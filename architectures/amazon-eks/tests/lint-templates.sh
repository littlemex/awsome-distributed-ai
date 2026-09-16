#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Mechanical checks for the amazon-eks CloudFormation deploy path.
#
#   bash architectures/amazon-eks/tests/lint-templates.sh
#
# Runs the ten checks frozen in the phase-2 interface document. Every check
# prints one line per problem saying which file failed, what was expected and
# what was found; the run ends with a count and exits non-zero if any check
# failed.
#
# Checks that need an AWS account (1 and 10) SKIP when no credentials are
# available, and checks that need `helm` (3 and 4) SKIP when helm is missing or
# cannot reach the chart repository, so the same script runs in CI (where the
# lint step runs before credentials are configured) and on a laptop. A SKIP is
# never counted as a pass: skips are counted and printed separately.
#
# Chart versions are read out of the templates, never hardcoded here — a lint
# that carries its own copy of a pinned version stops checking the moment the
# template moves on. If a version cannot be read, the check fails; it never
# falls back to a default.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # architectures/amazon-eks
cd "$ROOT"

PREREQ_T="assets/eks-cluster-prerequisites.yaml"
CLUSTER_T="assets/eks-cluster.yaml"
GPU_T="assets/eks-add-gpu-nodegroup.yaml"
ROOT_T="assets/eks-gpu-cluster-deploy-all.yaml"
ALL_T=("$PREREQ_T" "$CLUSTER_T" "$GPU_T" "$ROOT_T")
RENDER="tests/render-nic-block.py"
PARAMS_DOC="docs/PARAMETERS.md"

# Instance types the interface document requires NicLayout to carry, and the
# Kubernetes-visible values the templates and the test procedures share.
REQUIRED_NIC_TYPES=(
  g7e.12xlarge g7e.24xlarge g7e.48xlarge
  p4d.24xlarge p4de.24xlarge
  p5.48xlarge p5en.48xlarge
  p6-b200.48xlarge p6-b300.48xlarge
  g6e.12xlarge g6e.48xlarge
  g5.12xlarge g4dn.8xlarge
)
NIC_KEYS=(Cards PrimaryEfa SecondaryDeviceIndex EfaInterfaces)
GPU_TAINT_KEY="nvidia.com/gpu"

FAILURES=0
SKIPS=0

fail() { printf 'FAIL  [%s] %s\n' "$1" "$2"; FAILURES=$((FAILURES + 1)); }
skip() { printf 'SKIP  [%s] %s\n' "$1" "$2"; SKIPS=$((SKIPS + 1)); }
pass() { printf 'PASS  [%s] %s\n' "$1" "$2"; }
head_() { printf '\n=== check %s: %s\n' "$1" "$2"; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --------------------------------------------------------------------------
# Interpreter discovery. The YAML checks need PyYAML; the text checks do not.
# --------------------------------------------------------------------------
PY_ANY=""
for cand in "${PYTHON:-python3}" python3 python; do
  [ -n "$cand" ] || continue
  if command -v "$cand" >/dev/null 2>&1; then PY_ANY="$cand"; break; fi
done
PY_YAML=""
for cand in "${PYTHON:-python3}" python3 python3.13 python3.12 python3.11 python; do
  [ -n "$cand" ] || continue
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" -c 'import yaml' >/dev/null 2>&1; then PY_YAML="$cand"; break; fi
done

CFNQ="$TMP/cfnq.py"
TEXTQ="$TMP/textq.py"

# --------------------------------------------------------------------------
# cfnq.py — reads a CloudFormation template. PyYAML's SafeLoader rejects the
# short intrinsic tags (!Ref, !If, ...), so the loader below accepts every
# "!Something" tag and keeps its argument. Exit 3 = no PyYAML, 4 = the section
# the caller asked for is missing.
# --------------------------------------------------------------------------
cat > "$CFNQ" <<'PYEOF'
import sys

try:
    import yaml
except ImportError:
    sys.exit(3)


class CfnLoader(yaml.SafeLoader):
    pass


def _keep_tag(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {tag_suffix if tag_suffix == "Ref" else "Fn::" + tag_suffix: value}


CfnLoader.add_multi_constructor("!", _keep_tag)


def load(path):
    with open(path, encoding="utf-8") as handle:
        doc = yaml.load(handle, Loader=CfnLoader)
    if not isinstance(doc, dict):
        sys.stderr.write("%s: not a YAML mapping\n" % path)
        sys.exit(4)
    return doc


def section(doc, path, name):
    part = doc.get(name)
    if not isinstance(part, dict):
        sys.stderr.write("%s: no %s section\n" % (path, name))
        sys.exit(4)
    return part


cmd = sys.argv[1]
path = sys.argv[2]
doc = load(path)

if cmd == "params":
    for name in section(doc, path, "Parameters"):
        print(name)

elif cmd == "allowed":
    param = section(doc, path, "Parameters").get(sys.argv[3])
    if not isinstance(param, dict):
        sys.stderr.write("%s: no parameter %s\n" % (path, sys.argv[3]))
        sys.exit(4)
    values = param.get("AllowedValues")
    if not isinstance(values, list):
        sys.stderr.write("%s: parameter %s has no AllowedValues list\n" % (path, sys.argv[3]))
        sys.exit(4)
    for value in values:
        print(value)

elif cmd == "default":
    param = section(doc, path, "Parameters").get(sys.argv[3], {})
    print(param.get("Default", ""))

elif cmd == "resolve":
    # The literal this template gives a name, through the paths a version can
    # legitimately travel: a parameter default, or a CodeBuild environment
    # variable whose value is a Mappings lookup. Anything else (a Ref to another
    # parameter, a Sub, a nested intrinsic) is reported as unresolvable rather
    # than guessed, because the whole point of reading the template is to test
    # the version it really installs.
    name = sys.argv[3]

    def find_map(value):
        if not (isinstance(value, dict) and list(value) == ["Fn::FindInMap"]):
            return None
        keys = value["Fn::FindInMap"]
        if not (isinstance(keys, list) and len(keys) == 3):
            return None
        if not all(isinstance(k, str) for k in keys):
            return None                      # e.g. !FindInMap [M, !Ref P, K]
        table = (doc.get("Mappings") or {}).get(keys[0]) or {}
        entry = table.get(keys[1])
        if isinstance(entry, dict) and isinstance(entry.get(keys[2]), (str, int, float)):
            return str(entry[keys[2]])
        return None

    param = (doc.get("Parameters") or {}).get(name)
    if isinstance(param, dict) and param.get("Default") not in (None, ""):
        print(param["Default"])
        raise SystemExit(0)

    def walk(node):
        if isinstance(node, dict):
            variables = None
            environment = node.get("Environment")
            if isinstance(environment, dict):
                variables = environment.get("EnvironmentVariables")
            if isinstance(variables, list):
                for item in variables:
                    if not isinstance(item, dict) or item.get("Name") != name:
                        continue
                    value = item.get("Value")
                    if isinstance(value, (str, int, float)):
                        return str(value)
                    literal = find_map(value)
                    if literal is not None:
                        return literal
            for child in node.values():
                found = walk(child)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = walk(child)
                if found is not None:
                    return found
        return None

    literal = walk(doc.get("Resources") or {})
    if literal is None:
        sys.stderr.write("%s: cannot resolve %s to a literal\n" % (path, name))
        sys.exit(4)
    print(literal)

elif cmd == "mapping":
    mappings = section(doc, path, "Mappings")
    table = mappings.get(sys.argv[3])
    if not isinstance(table, dict):
        sys.stderr.write("%s: no mapping %s (present: %s)\n"
                         % (path, sys.argv[3], ", ".join(sorted(mappings)) or "none"))
        sys.exit(4)
    for key in sorted(table):
        entry = table[key]
        if not isinstance(entry, dict):
            print("%s\t!not-a-mapping\t%r\t%s" % (key, entry, type(entry).__name__))
            continue
        for sub in sorted(entry):
            print("%s\t%s\t%s\t%s" % (key, sub, entry[sub], type(entry[sub]).__name__))

elif cmd == "mapping-names":
    for name in sorted(doc.get("Mappings") or {}):
        print(name)

elif cmd == "outputs":
    for name in section(doc, path, "Outputs"):
        print(name)

else:
    sys.stderr.write("unknown command %s\n" % cmd)
    sys.exit(2)
PYEOF

# --------------------------------------------------------------------------
# textq.py — the checks that read the files as text, so they work without
# PyYAML: the generated NetworkInterfaces block, Markdown links and anchors,
# and the version-pinning scan. Every subcommand prints one problem per line
# and exits 1 when it found problems, 2 on a usage/structural error.
# --------------------------------------------------------------------------
cat > "$TEXTQ" <<'PYEOF'
import os
import re
import sys

BEGIN = re.compile(r"#\s*BEGIN\s+(?:GENERATED|generated)\b|#\s*BEGIN\b.*render-nic-block\.py", re.I)
END = re.compile(r"#\s*END\s+(?:GENERATED|generated)\b|#\s*END\b.*render-nic-block\.py", re.I)


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def nicblock(path):
    """Print the committed NetworkInterfaces block, with and without its key line.

    Two forms are printed because the interface document does not say whether
    render-nic-block.py emits the `NetworkInterfaces:` key or only the list
    under it; the caller accepts an exact match against either. Preferred is a
    marked region:

        # BEGIN render-nic-block.py
        ...
        # END render-nic-block.py
    """
    lines = read(path).splitlines()
    begin = [i for i, line in enumerate(lines) if BEGIN.search(line)]
    end = [i for i, line in enumerate(lines) if END.search(line)]
    if begin and end:
        if len(begin) != 1 or len(end) != 1 or end[0] < begin[0]:
            sys.stderr.write("%s: render-nic-block.py BEGIN/END markers are not a single "
                             "ordered pair (BEGIN at %s, END at %s)\n"
                             % (path, begin, end))
            return 2
        block = lines[begin[0] + 1:end[0]]
        sys.stdout.write("MARKED\n")
    else:
        starts = [i for i, line in enumerate(lines)
                  if re.match(r"^\s*NetworkInterfaces:\s*$", line)]
        if not starts:
            sys.stderr.write("%s: no `NetworkInterfaces:` key and no render-nic-block.py "
                             "BEGIN/END markers found\n" % path)
            return 2
        if len(starts) > 1:
            sys.stderr.write("%s: `NetworkInterfaces:` appears %d times (lines %s) — the "
                             "comparison needs one block; mark the generated region with "
                             "`# BEGIN render-nic-block.py` / `# END render-nic-block.py`\n"
                             % (path, len(starts), [i + 1 for i in starts]))
            return 2
        start = starts[0]
        indent = len(lines[start]) - len(lines[start].lstrip())
        block = [lines[start]]
        for line in lines[start + 1:]:
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            block.append(line)
        while block and not block[-1].strip():
            block.pop()
        sys.stdout.write("KEYED\n")
    with open(sys.argv[3], "w", encoding="utf-8") as handle:      # with the key line
        handle.write("\n".join(block) + "\n")
    body = block[1:] if re.match(r"^\s*NetworkInterfaces:\s*$", block[0]) else block
    with open(sys.argv[4], "w", encoding="utf-8") as handle:      # without the key line
        handle.write("\n".join(body) + "\n")
    return 0


def slugs(path):
    """GitHub heading anchors for a Markdown file, including duplicate suffixes."""
    out = []
    seen = {}
    in_fence = False
    for line in read(path).splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.startswith("#"):
            continue
        text = re.sub(r"^#{1,6}\s+", "", line)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)      # links -> their text
        text = text.replace("`", "").replace("*", "").replace("_", "_")
        slug = re.sub(r"[^\w\- ]", "", text.lower(), flags=re.UNICODE).strip()
        slug = re.sub(r"\s+", "-", slug)
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        out.append(slug if count == 0 else "%s-%d" % (slug, count))
    return out


LINK = re.compile(r"(?<!\\)\[[^\]]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")


def links(paths):
    problems = 0
    cache = {}
    for path in paths:
        base = os.path.dirname(path)
        in_fence = False
        for number, line in enumerate(read(path).splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for target in LINK.findall(line):
                if re.match(r"^(https?:|mailto:|ftp:|tel:)", target) or "${" in target:
                    continue
                relpath, _, anchor = target.partition("#")
                if relpath:
                    resolved = os.path.normpath(os.path.join(base, relpath))
                    if not os.path.exists(resolved):
                        print("%s:%d: link target `%s` resolves to %s, which does not exist"
                              % (path, number, target, resolved))
                        problems += 1
                        continue
                else:
                    resolved = path
                if not anchor:
                    continue
                if not resolved.endswith(".md"):
                    continue
                if resolved not in cache:
                    cache[resolved] = slugs(resolved)
                if anchor.lower() not in cache[resolved]:
                    print("%s:%d: anchor `#%s` has no matching heading in %s"
                          % (path, number, anchor, resolved))
                    problems += 1
    return 1 if problems else 0


UNPINNED = [
    (re.compile(r":latest(?![\w.-])"), "image or chart reference tagged `latest`"),
    (re.compile(r"/latest(?=[/\"'\s]|$)"), "`latest` in a download path — pin the version"),
    (re.compile(r"stable\.txt"), "version resolved from `stable.txt` at run time — pin it"),
    (re.compile(r"--version[= ]\s*(latest|stable)"), "`--version latest` is not a pin"),
]
IMAGE_KEY = re.compile(r"^\s*[Ii]mage:\s*(\S.*?)\s*$")
HELM_INSTALL = re.compile(r"helm\s+(?:upgrade|install)\b")


def pins(paths):
    problems = 0
    for path in paths:
        raw = read(path).splitlines()
        # Join backslash continuations so a helm command split over lines is
        # scanned as one command, keeping the first line's number.
        logical = []
        buffer, first = "", None
        for number, line in enumerate(raw, 1):
            if first is None:
                first = number
            stripped = line.rstrip()
            if stripped.endswith("\\"):
                buffer += stripped[:-1] + " "
                continue
            logical.append((first, buffer + stripped))
            buffer, first = "", None
        if buffer:
            logical.append((first, buffer))
        for number, line in logical:
            for pattern, why in UNPINNED:
                if pattern.search(line):
                    print("%s:%d: %s: %s" % (path, number, why, line.strip()[:160]))
                    problems += 1
            if HELM_INSTALL.search(line) and "--version" not in line:
                print("%s:%d: `helm upgrade/install` without `--version`: %s"
                      % (path, number, line.strip()[:160]))
                problems += 1
            found = IMAGE_KEY.match(line)
            if found:
                value = found.group(1)
                if value.startswith(("!", "{", "$", "'${", '"${')) or "${" in value:
                    continue
                bare = value.strip("'\"")
                if "@sha256:" in bare:
                    continue
                tail = bare.rsplit("/", 1)[-1]
                if ":" not in tail:
                    print("%s:%d: image `%s` has no tag or digest" % (path, number, bare))
                    problems += 1
    return 1 if problems else 0


cmd = sys.argv[1]
if cmd == "nicblock":
    sys.exit(nicblock(sys.argv[2]))
elif cmd == "links":
    sys.exit(links(sys.argv[2:]))
elif cmd == "pins":
    sys.exit(pins(sys.argv[2:]))
sys.stderr.write("unknown command %s\n" % cmd)
sys.exit(2)
PYEOF

printf 'lint-templates.sh — %s\n' "$ROOT"

# --------------------------------------------------------------------------
# check 0 — the files the other checks read
# --------------------------------------------------------------------------
head_ 0 "expected files are present"
for f in "${ALL_T[@]}" "$RENDER" "$PARAMS_DOC" README.md; do
  if [ -f "$f" ]; then pass 0 "$f"; else fail 0 "$f: expected to exist, not found"; fi
done
if [ -z "$PY_ANY" ]; then
  fail 0 "no python3 on PATH: checks 2-10 cannot run"
elif [ -z "$PY_YAML" ]; then
  fail 0 "no python3 with PyYAML on PATH (install: python3 -m pip install pyyaml): the checks that parse the templates cannot run"
else
  pass 0 "$PY_YAML can import yaml"
fi

cfnq() { "$PY_YAML" "$CFNQ" "$@"; }

# --------------------------------------------------------------------------
# check 1 — aws cloudformation validate-template on all four templates
# --------------------------------------------------------------------------
head_ 1 "aws cloudformation validate-template on all four templates"
HAVE_AWS=no
if ! command -v aws >/dev/null 2>&1; then
  skip 1 "aws CLI not on PATH"
elif ! aws sts get-caller-identity >/dev/null 2>&1; then
  skip 1 "no usable AWS credentials (validate-template is an authenticated call)"
else
  HAVE_AWS=yes
fi
if [ "$HAVE_AWS" = yes ]; then
  for t in "${ALL_T[@]}"; do
    [ -f "$t" ] || { skip 1 "$t: file missing"; continue; }
    size=$(wc -c < "$t" | tr -d ' ')
    if [ "$size" -gt 51200 ]; then
      skip 1 "$t: $size bytes exceeds the 51,200-byte --template-body limit; validate with --template-url from S3 (docs/DEPLOY-TESTING.md)"
      continue
    fi
    if out=$(aws cloudformation validate-template --template-body "file://$t" 2>&1); then
      pass 1 "$t"
    else
      fail 1 "$t: validate-template rejected it: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)"
    fi
  done
fi

# --------------------------------------------------------------------------
# check 2 — GpuInstanceType.AllowedValues, NicLayout and GpuCount agree
# --------------------------------------------------------------------------
head_ 2 "every selectable GPU type is in NicLayout and GpuCount, with all keys"
ALLOWED_FILE="$TMP/allowed.txt"
: > "$ALLOWED_FILE"
if [ -z "$PY_YAML" ] || [ ! -f "$GPU_T" ] || [ ! -f "$ROOT_T" ]; then
  skip 2 "needs PyYAML, $GPU_T and $ROOT_T"
else
  gpu_allowed=$(cfnq allowed "$GPU_T" GpuInstanceType 2>"$TMP/e2a") || true
  if [ -z "$gpu_allowed" ]; then
    fail 2 "$GPU_T: could not read GpuInstanceType.AllowedValues: $(tr '\n' ' ' < "$TMP/e2a")"
  fi
  root_allowed=$(cfnq allowed "$ROOT_T" GpuInstanceType 2>"$TMP/e2b") || true
  if [ -z "$root_allowed" ]; then
    fail 2 "$ROOT_T: could not read GpuInstanceType.AllowedValues: $(tr '\n' ' ' < "$TMP/e2b")"
  fi
  if [ -n "$gpu_allowed" ] && [ -n "$root_allowed" ] && [ "$gpu_allowed" != "$root_allowed" ]; then
    fail 2 "GpuInstanceType.AllowedValues differs between the root and the GPU template: root has [$(echo "$root_allowed" | tr '\n' ' ')], gpu has [$(echo "$gpu_allowed" | tr '\n' ' ')] — the root passes the value down, so a type the root allows and the GPU template does not is a deploy-time failure"
  fi
  printf '%s\n' "$gpu_allowed" | sed '/^$/d' > "$ALLOWED_FILE"

  for table in NicLayout GpuCount; do
    if ! cfnq mapping "$GPU_T" "$table" > "$TMP/$table.tsv" 2>"$TMP/e2$table"; then
      fail 2 "$GPU_T: mapping $table could not be read: $(tr '\n' ' ' < "$TMP/e2$table")"
      : > "$TMP/$table.tsv"
    fi
  done

  nic_types=$(cut -f1 "$TMP/NicLayout.tsv" | sort -u)
  gpucount_types=$(cut -f1 "$TMP/GpuCount.tsv" | sort -u)

  while read -r t; do
    [ -n "$t" ] || continue
    grep -qx "$t" <<<"$nic_types" || fail 2 "$GPU_T: GpuInstanceType.AllowedValues offers '$t' but NicLayout has no entry for it (the launch template would render with no NIC layout)"
    grep -qx "$t" <<<"$gpucount_types" || fail 2 "$GPU_T: GpuInstanceType.AllowedValues offers '$t' but GpuCount has no entry for it (the bootstrap cannot assert an exact nvidia.com/gpu count)"
  done < "$ALLOWED_FILE"

  for t in "${REQUIRED_NIC_TYPES[@]}"; do
    grep -qx "$t" <<<"$nic_types" || fail 2 "$GPU_T: NicLayout is missing '$t', which the frozen interface lists as a required entry"
  done

  if [ -n "$nic_types" ] && [ -n "$gpucount_types" ] && [ "$nic_types" != "$gpucount_types" ]; then
    fail 2 "$GPU_T: NicLayout and GpuCount cover different instance types: only in NicLayout [$(comm -23 <(echo "$nic_types") <(echo "$gpucount_types") | tr '\n' ' ')], only in GpuCount [$(comm -13 <(echo "$nic_types") <(echo "$gpucount_types") | tr '\n' ' ')]"
  fi

  while read -r t; do
    [ -n "$t" ] || continue
    for key in "${NIC_KEYS[@]}"; do
      line=$(awk -F'\t' -v t="$t" -v k="$key" '$1==t && $2==k {print; exit}' "$TMP/NicLayout.tsv")
      if [ -z "$line" ]; then
        fail 2 "$GPU_T: NicLayout[$t] has no '$key' key (all four keys are required)"
        continue
      fi
      value=$(printf '%s' "$line" | cut -f3)
      vtype=$(printf '%s' "$line" | cut -f4)
      [ "$vtype" = str ] || fail 2 "$GPU_T: NicLayout[$t][$key] is $vtype '$value'; the interface freezes these as strings — quote it (FindInMap returns a string either way, but an unquoted value invites Fn::If comparisons that never match)"
    done
    for key in Cards SecondaryDeviceIndex EfaInterfaces; do
      value=$(awk -F'\t' -v t="$t" -v k="$key" '$1==t && $2==k {print $3; exit}' "$TMP/NicLayout.tsv")
      [ -n "$value" ] || continue
      case "$value" in
        ''|*[!0-9]*) fail 2 "$GPU_T: NicLayout[$t][$key] is '$value'; expected a whole number" ;;
      esac
    done
    primary=$(awk -F'\t' -v t="$t" '$1==t && $2=="PrimaryEfa" {print $3; exit}' "$TMP/NicLayout.tsv")
    case "${primary:-unset}" in
      true|false|unset) : ;;
      *) fail 2 "$GPU_T: NicLayout[$t][PrimaryEfa] is '$primary'; expected \"true\" or \"false\"" ;;
    esac
    devidx=$(awk -F'\t' -v t="$t" '$1==t && $2=="SecondaryDeviceIndex" {print $3; exit}' "$TMP/NicLayout.tsv")
    case "${devidx:-unset}" in
      0|1|unset) : ;;
      *) fail 2 "$GPU_T: NicLayout[$t][SecondaryDeviceIndex] is '$devidx'; only 0 and 1 have been run on hardware" ;;
    esac
  done <<<"$nic_types"

  while read -r t; do
    [ -n "$t" ] || continue
    keys=$(awk -F'\t' -v t="$t" '$1==t {print $2}' "$TMP/GpuCount.tsv" | sort | tr '\n' ',' | sed 's/,$//')
    [ "$keys" = "Gpus" ] || fail 2 "$GPU_T: GpuCount[$t] has keys [$keys]; the interface freezes one key, 'Gpus'"
    value=$(awk -F'\t' -v t="$t" '$1==t && $2=="Gpus" {print $3; exit}' "$TMP/GpuCount.tsv")
    case "${value:-x}" in
      ''|*[!0-9]*) fail 2 "$GPU_T: GpuCount[$t][Gpus] is '$value'; expected a whole number of GPUs" ;;
      0) fail 2 "$GPU_T: GpuCount[$t][Gpus] is 0; a GPU node group with no GPUs cannot pass the readiness check" ;;
    esac
  done <<<"$gpucount_types"
  pass 2 "read $(wc -l < "$ALLOWED_FILE" | tr -d ' ') selectable type(s), $(printf '%s' "$nic_types" | grep -c . || true) NicLayout entr(y|ies)"
fi

# --------------------------------------------------------------------------
# chartver.py — reads the pinned chart version and the chart repository URL
# out of the templates, so checks 3 and 4 test what the templates install
# rather than what this script remembers.
# --------------------------------------------------------------------------
CHARTVER="$TMP/chartver.py"
export CFNQ CFNQ_PY
CFNQ_PY="${PY_YAML:-$PY_ANY}"
cat > "$CHARTVER" <<'PYEOF'
import os
import re
import sys

VERSION = re.compile(r"--version[= ]\s*[\"']?"
                     r"(\$\{!?[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*|[A-Za-z0-9][A-Za-z0-9.\-+]*)")
REPO_ADD = re.compile(r"helm\s+repo\s+add\s+(\S+)\s+(\S+)")


def logical(path):
    raw = open(path, encoding="utf-8").read().splitlines()
    out, buffer, first = [], "", None
    for number, line in enumerate(raw, 1):
        if first is None:
            first = number
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        out.append((first, buffer + stripped))
        buffer, first = "", None
    if buffer:
        out.append((first, buffer))
    return raw, out


cmd, chart, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
versions, urls, aliases = [], [], set()
for path in paths:
    try:
        raw, lines = logical(path)
    except OSError:
        continue
    for number, line in lines:
        if chart not in line:
            continue
        for alias in re.findall(r"([A-Za-z0-9_.\-]+)/" + re.escape(chart), line):
            aliases.add(alias)
        found = VERSION.findall(line)
        if not found:
            lo = max(0, number - 9)
            hi = min(len(raw), number + 8)
            for near in raw[lo:hi]:
                found += VERSION.findall(near)
        versions += found
    for number, line in lines:
        alias_url = REPO_ADD.search(line)
        if alias_url:
            urls.append((alias_url.group(1), alias_url.group(2)))

INSTALL = re.compile(r"helm\s+(?:upgrade|install)\b")
PLACEHOLDER = re.compile(r"\$\{!?([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def lookup(name, paths):
    """The value the templates give a variable: shell assignment, buildspec key, or default."""
    for path in paths:
        try:
            body = open(path, encoding="utf-8").read()
        except OSError:
            continue
        for pattern in (r"(?:^|[\s;\"'])(?:export\s+)?" + re.escape(name) + r"=[\"']?([^\s\"']+)",
                        r"(?:^|\s)" + re.escape(name) + r":\s*[\"']?([^\s\"']+)"):
            found = re.search(pattern, body, re.M)
            if found:
                return found.group(1)
    # The CloudFormation-side paths (parameter default, CodeBuild environment
    # variable, Mappings lookup) live in cfnq.py so there is one implementation
    # of them; this script is handed its location.
    helper = os.environ.get("CFNQ")
    if helper:
        import subprocess
        interpreter = os.environ.get("CFNQ_PY") or sys.executable
        for path in paths:
            done = subprocess.run([interpreter, helper, "resolve", path, name],
                                  capture_output=True, text=True)
            if done.returncode == 0 and done.stdout.strip():
                return done.stdout.strip()
    return None


def setargs(chart, paths):
    """Print the --set family arguments the templates pass when installing `chart`.

    The point is that check 4 must render what the template installs, not what the
    chart defaults to: a template that drops the GPU toleration has to make the
    check fail. Anything that cannot be reproduced exactly is reported instead of
    being replaced by a default.
    """
    command = None
    for path in paths:
        try:
            _, lines = logical(path)
        except OSError:
            continue
        for _, line in lines:
            if chart in line and INSTALL.search(line):
                command = line
                break
        if command:
            break
    if command is None:
        print("NOTFOUND\tno `helm upgrade/install` command for %s in the templates" % chart)
        return 1
    for flag in ("-f ", "--values"):
        if flag.strip() in command.split():
            print("VALUESFILE\t%s passes a values file (%s), which this lint cannot reproduce; "
                  "express overrides as --set/--set-json so they can be checked" % (chart, flag.strip()))
            return 1
    try:
        import shlex
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        tokens = re.findall(r"--set(?:-json|-string|-literal|-file)?[= ]\s*"
                            r"(?:\"[^\"]*\"|'[^']*'|\S+)", command)
        tokens = [part for token in tokens for part in token.replace("=", " ", 1).split(None, 1)]
    pairs, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--set"):
            if "=" in token and token.split("=", 1)[0] in (
                    "--set", "--set-json", "--set-string", "--set-literal", "--set-file"):
                flag, value = token.split("=", 1)
            else:
                flag, value = token, tokens[index + 1] if index + 1 < len(tokens) else ""
                index += 1
            pairs.append((flag, value))
        index += 1
    for flag, value in pairs:
        unresolved = []

        def substitute(match):
            name = match.group(1) or match.group(2)
            found = lookup(name, paths)
            if found is None:
                unresolved.append(name)
                return "lintunresolved"
            return found

        resolved = PLACEHOLDER.sub(substitute, value)
        for name in unresolved:
            print("UNRESOLVED\t%s\t%s\t%s" % (name, flag, value))
        print("SET\t%s\t%s" % (flag, resolved))
    return 0


if cmd == "setargs":
    sys.exit(setargs(chart, paths))
elif cmd == "version":
    for value in sorted(set(versions)):
        print(value)
elif cmd == "repourl":
    wanted = {url for alias, url in urls if alias in aliases} or {url for _, url in urls}
    for value in sorted(wanted):
        print(value.strip("\"'"))
PYEOF

resolve_version_token() {   # a literal version, or the value the template gives the variable
  local token="$1" name value t hop=0
  while [ "$hop" -lt 4 ]; do
    case "$token" in
      [vV][0-9]*|[0-9]*) printf '%s\n' "$token"; return 0 ;;
    esac
    name=$(printf '%s' "$token" | sed -E 's/^\$\{!?//; s/\}$//; s/^\$//')
    value=""
    for t in "${ALL_T[@]}"; do
      [ -f "$t" ] || continue
      # a shell assignment (UserData, buildspec commands) ...
      value=$(grep -oE "(^|[[:space:];'\"])(export[[:space:]]+)?${name}=[\"']?[^[:space:]\"']+" "$t" 2>/dev/null | head -1 | sed -E "s/.*${name}=[\"']?//" || true)
      [ -n "$value" ] && break
      # ... or a YAML key, which is how a CodeBuild buildspec declares env variables
      value=$(grep -oE "(^|[[:space:]])${name}:[[:space:]]*[\"']?[^[:space:]\"']+" "$t" 2>/dev/null | head -1 | sed -E "s/.*${name}:[[:space:]]*[\"']?//" || true)
      [ -n "$value" ] && break
    done
    if [ -z "$value" ] && [ -n "$PY_YAML" ]; then
      # a CloudFormation parameter default, or a CodeBuild environment variable
      # whose value is a Mappings lookup (env -> FindInMap -> Mappings)
      for t in "${ALL_T[@]}"; do
        [ -f "$t" ] || continue
        value=$(cfnq resolve "$t" "$name" 2>/dev/null || true)
        [ -n "$value" ] && break
      done
    fi
    if [ -z "$value" ]; then
      printf 'UNRESOLVED:%s\n' "$name"
      return 0
    fi
    token="$value"
    hop=$((hop + 1))
  done
  printf 'UNRESOLVED:%s\n' "$name"
}

chart_helm_args() {   # $1 = chart name, $2 = fallback repo URL
  local chart="$1" default_url="$2" tokens url token resolved="" one
  tokens=$("$PY_ANY" "$CHARTVER" version "$chart" "${ALL_T[@]}" 2>/dev/null || true)
  if [ -z "$tokens" ]; then
    printf 'NOVERSION\n'
    return 0
  fi
  while read -r token; do
    [ -n "$token" ] || continue
    one=$(resolve_version_token "$token")
    case "$one" in
      UNRESOLVED:*) printf 'UNRESOLVED %s\n' "${one#UNRESOLVED:}"; return 0 ;;
    esac
    resolved="${resolved}${one}
"
  done <<<"$tokens"
  resolved=$(printf '%s' "$resolved" | sed '/^$/d' | sort -u)
  if [ "$(printf '%s\n' "$resolved" | grep -c .)" -gt 1 ]; then
    printf 'AMBIGUOUS %s\n' "$(printf '%s' "$resolved" | tr '\n' ' ')"
    return 0
  fi
  url=$("$PY_ANY" "$CHARTVER" repourl "$chart" "${ALL_T[@]}" 2>/dev/null | head -1 || true)
  [ -n "$url" ] || url="$default_url"
  printf 'OK %s %s\n' "$resolved" "$url"
}

render_chart() {   # $1 = release, $2 = namespace, $3 = chart, $4 = version, $5 = url, $6 = out file,
                   # remaining arguments are passed to helm template as-is
  local release="$1" ns="$2" chart="$3" version="$4" url="$5" out="$6"
  shift 6
  helm template "$release" "$chart" --repo "$url" --version "$version" -n "$ns" "$@" \
    > "$out" 2>"$out.err"
}

# Collects the --set family arguments the templates pass when installing a chart, so a
# render reflects what will be installed rather than the chart's defaults. Sets SETARGS.
# Returns 1 when the arguments cannot be reproduced exactly; the caller must fail then,
# because rendering with defaults would let a template drop a toleration unnoticed.
collect_setargs() {   # $1 = chart name, $2 = check number
  local chart="$1" check="$2" kind a b c reason
  SETARGS=()
  if ! "${PY_YAML:-$PY_ANY}" "$CHARTVER" setargs "$chart" "${ALL_T[@]}" \
       > "$TMP/setargs-$chart.txt" 2>"$TMP/setargs-$chart.err"; then
    reason=$(cut -f2- "$TMP/setargs-$chart.txt" | tr '\n' ' ')
    [ -n "$reason" ] || reason=$(tr '\n' ' ' < "$TMP/setargs-$chart.err")
    fail "$check" "$reason"
    return 1
  fi
  local bad=0
  while IFS=$'\t' read -r kind a b c; do
    case "$kind" in
      SET) SETARGS+=("$a" "$b") ;;
      UNRESOLVED)
        if printf '%s %s' "$b" "$c" | grep -qi 'toleration\|taint'; then
          fail "$check" "$chart is installed with '$b $c', and the value of \$$a cannot be read from the templates; this argument decides the tolerations, so the render would not be the one the template installs"
          bad=1
        else
          printf 'NOTE  [%s] %s: the value of $%s in '"'"'%s %s'"'"' is not readable from the templates; rendered with a placeholder (it does not touch tolerations)\n' \
            "$check" "$chart" "$a" "$b" "$c"
        fi ;;
    esac
  done < "$TMP/setargs-$chart.txt"
  return "$bad"
}

# A failed `helm template` has two very different causes, and treating them the
# same would let a pinned version that does not exist pass as "offline". The
# error text separates them: a missing chart or version is a failure, an
# unreachable repository is a skip.
helm_error_kind() {   # $1 = the .err file written by render_chart
  if grep -qiE 'no cached repo|not found|no chart (version|name) found|could not find|improper constraint' "$1"; then
    printf 'MISSING\n'
  elif grep -qiE 'dial tcp|no such host|timeout|timed out|connection refused|deadline exceeded|x509|temporary failure in name resolution|tls' "$1"; then
    printf 'OFFLINE\n'
  else
    printf 'UNKNOWN\n'
  fi
}

# --------------------------------------------------------------------------
# check 3 — the pinned EFA chart schedules on every selectable GPU type
# --------------------------------------------------------------------------
head_ 3 "every selectable GPU type is in the pinned EFA chart's nodeAffinity list"
EFA_CHART=aws-efa-k8s-device-plugin
EFA_DEFAULT_URL=https://aws.github.io/eks-charts
if ! command -v helm >/dev/null 2>&1; then
  skip 3 "helm not on PATH (install helm, or run: helm repo add eks $EFA_DEFAULT_URL)"
elif [ ! -s "$ALLOWED_FILE" ]; then
  skip 3 "GpuInstanceType.AllowedValues could not be read (see check 2)"
else
  read -r status version url <<<"$(chart_helm_args "$EFA_CHART" "$EFA_DEFAULT_URL")"
  case "$status" in
    NOVERSION)
      fail 3 "no '--version' pin for $EFA_CHART found in ${ALL_T[*]}: the chart version must be readable from the template, and this check must not guess one" ;;
    AMBIGUOUS)
      fail 3 "several different chart versions appear next to $EFA_CHART in the templates ($version $url) — pin one" ;;
    UNRESOLVED)
      fail 3 "$EFA_CHART is installed with --version \$$version, and this lint cannot resolve '$version' to a literal version from the templates (looked for a shell assignment and a CloudFormation parameter default); it will not guess, so pin the chart version where the lint can read it" ;;
    OK)
      if render_chart "$EFA_CHART" kube-system "$EFA_CHART" "$version" "$url" "$TMP/efa.yaml"; then
        grep -oE '^[[:space:]]*-[[:space:]]+[a-z0-9]+[.-][a-z0-9.]+' "$TMP/efa.yaml" \
          | sed -E 's/^[[:space:]]*-[[:space:]]+//' | sort -u > "$TMP/efa-types.txt"
        while read -r t; do
          [ -n "$t" ] || continue
          if grep -qx "$t" "$TMP/efa-types.txt"; then continue; fi
          efa_ifaces=$(awk -F'\t' -v t="$t" '$1==t && $2=="EfaInterfaces" {print $3; exit}' "$TMP/NicLayout.tsv" 2>/dev/null || true)
          if [ "${efa_ifaces:-}" = "0" ]; then
            skip 3 "$t is absent from $EFA_CHART $version's instance-type list, but NicLayout says EfaInterfaces=0, so the EFA plugin is not meant to run there — see the deviation note in the header of this check"
          else
            fail 3 "$t is a selectable GpuInstanceType but is absent from $EFA_CHART $version's nodeAffinity instance-type list, so the EFA device plugin would never schedule on those nodes (bump the chart, or drop the type)"
          fi
        done < "$ALLOWED_FILE"
        pass 3 "$EFA_CHART $version lists $(wc -l < "$TMP/efa-types.txt" | tr -d ' ') instance types"
      else
        case "$(helm_error_kind "$TMP/efa.yaml.err")" in
          MISSING) fail 3 "$EFA_CHART $version does not exist in $url — the version the template pins cannot be installed: $(head -2 "$TMP/efa.yaml.err" | tr '\n' ' ')" ;;
          OFFLINE) skip 3 "cannot reach $url to render $EFA_CHART $version (run: helm repo update): $(head -2 "$TMP/efa.yaml.err" | tr '\n' ' ')" ;;
          *)       fail 3 "helm template failed for $EFA_CHART $version from $url for a reason this lint cannot classify: $(head -3 "$TMP/efa.yaml.err" | tr '\n' ' ')" ;;
        esac
      fi ;;
  esac
fi

# --------------------------------------------------------------------------
# check 4 — every DaemonSet the pinned NVIDIA chart renders tolerates the taint
# --------------------------------------------------------------------------
head_ 4 "the pinned NVIDIA chart's DaemonSets tolerate $GPU_TAINT_KEY:NoSchedule"
NVDP_CHART=nvidia-device-plugin
NVDP_DEFAULT_URL=https://nvidia.github.io/k8s-device-plugin
if ! command -v helm >/dev/null 2>&1; then
  skip 4 "helm not on PATH (install helm, or run: helm repo add nvdp $NVDP_DEFAULT_URL)"
else
  read -r status version url <<<"$(chart_helm_args "$NVDP_CHART" "$NVDP_DEFAULT_URL")"
  case "$status" in
    NOVERSION)
      fail 4 "no '--version' pin for $NVDP_CHART found in ${ALL_T[*]}" ;;
    AMBIGUOUS)
      fail 4 "several different chart versions appear next to $NVDP_CHART in the templates ($version $url) — pin one" ;;
    UNRESOLVED)
      fail 4 "$NVDP_CHART is installed with --version \$$version, and this lint cannot resolve '$version' to a literal version from the templates (looked for a shell assignment and a CloudFormation parameter default); it will not guess, so pin the chart version where the lint can read it" ;;
    OK)
      if ! collect_setargs "$NVDP_CHART" 4; then
        :   # collect_setargs already reported why the render cannot be trusted
      elif render_chart nvdp nvidia-device-plugin "$NVDP_CHART" "$version" "$url" \
           "$TMP/nvdp.yaml" ${SETARGS[@]+"${SETARGS[@]}"}; then
        if [ ${#SETARGS[@]} -gt 0 ]; then
          pass 4 "$NVDP_CHART $version rendered with the template's own arguments: ${SETARGS[*]}"
        else
          pass 4 "$NVDP_CHART $version: the template passes no --set arguments, so the chart defaults are what gets installed"
        fi
        "$PY_ANY" - "$TMP/nvdp.yaml" "$GPU_TAINT_KEY" <<'PYEOF' > "$TMP/nvdp-report.txt" || true
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
key = sys.argv[2]
for document in text.split("\n---\n"):
    if not re.search(r"^kind:\s*DaemonSet\s*$", document, re.M):
        continue
    name = re.search(r"^\s{2}name:\s*(\S+)", document, re.M)
    name = name.group(1) if name else "<unnamed>"
    block = re.search(r"tolerations:\n((?:\s+[-\s].*\n)+)", document)
    tolerated = False
    if block:
        for item in re.split(r"\n\s+-\s", "\n" + block.group(1)):
            if not item.strip():
                continue
            has_key = re.search(r"key:\s*[\"']?" + re.escape(key), item)
            effect = re.search(r"effect:\s*(\S+)", item)
            operator = re.search(r"operator:\s*(\S+)", item)
            if has_key and (effect is None or effect.group(1) == "NoSchedule"):
                tolerated = True
            elif operator and operator.group(1) == "Exists" and has_key is None and effect is None:
                tolerated = True      # tolerates everything
    print("%s\t%s" % ("OK" if tolerated else "MISSING", name))
PYEOF
        if [ ! -s "$TMP/nvdp-report.txt" ]; then
          fail 4 "$NVDP_CHART $version rendered no DaemonSet at all — the device plugin is a DaemonSet, so the render or the pin is wrong"
        fi
        while IFS=$'\t' read -r verdict name; do
          if [ "$verdict" = OK ]; then
            pass 4 "$name tolerates $GPU_TAINT_KEY:NoSchedule"
          else
            fail 4 "$NVDP_CHART $version renders DaemonSet '$name' with no toleration for $GPU_TAINT_KEY:NoSchedule; the GPU nodes carry that taint, so its pods would stay Pending and no node would advertise nvidia.com/gpu"
          fi
        done < "$TMP/nvdp-report.txt"
      else
        case "$(helm_error_kind "$TMP/nvdp.yaml.err")" in
          MISSING) fail 4 "$NVDP_CHART $version does not exist in $url — the version the template pins cannot be installed: $(head -2 "$TMP/nvdp.yaml.err" | tr '\n' ' ')" ;;
          OFFLINE) skip 4 "cannot reach $url to render $NVDP_CHART $version (run: helm repo update): $(head -2 "$TMP/nvdp.yaml.err" | tr '\n' ' ')" ;;
          *)       fail 4 "helm template failed for $NVDP_CHART $version from $url for a reason this lint cannot classify: $(head -3 "$TMP/nvdp.yaml.err" | tr '\n' ' ')" ;;
        esac
      fi ;;
  esac
fi

# --------------------------------------------------------------------------
# check 5 — the committed NetworkInterfaces block is what the generator emits
# --------------------------------------------------------------------------
head_ 5 "$RENDER output equals the NetworkInterfaces block in $GPU_T"
if [ ! -f "$RENDER" ]; then
  fail 5 "$RENDER: expected to exist (the committed block is generated by it); nothing to compare against"
elif [ ! -f "$GPU_T" ]; then
  skip 5 "$GPU_T missing"
elif [ -z "$PY_ANY" ]; then
  skip 5 "no python3 on PATH"
else
  if ! "$PY_ANY" "$RENDER" > "$TMP/render.out" 2>"$TMP/render.err"; then
    fail 5 "$RENDER exited non-zero: $(head -3 "$TMP/render.err" | tr '\n' ' ')"
  elif [ ! -s "$TMP/render.out" ]; then
    fail 5 "$RENDER produced no output"
  else
    if mode=$("$PY_ANY" "$TEXTQ" nicblock "$GPU_T" "$TMP/block-keyed.txt" "$TMP/block-body.txt" 2>"$TMP/nic.err"); then
      if cmp -s "$TMP/render.out" "$TMP/block-body.txt"; then
        pass 5 "byte-identical to the committed block ($mode, key line excluded)"
      elif cmp -s "$TMP/render.out" "$TMP/block-keyed.txt"; then
        pass 5 "byte-identical to the committed block ($mode, key line included)"
      else
        fail 5 "$GPU_T: the committed NetworkInterfaces block is not what $RENDER emits — regenerate it (the diff below is the committed block against the render; < is the render)"
        diff "$TMP/render.out" "$TMP/block-body.txt" | sed 's/^/        /' | head -40
      fi
    else
      fail 5 "$GPU_T: could not locate the generated block: $(tr '\n' ' ' < "$TMP/nic.err")"
    fi
  fi
fi

# --------------------------------------------------------------------------
# check 6 — the root template's parameters and $PARAMS_DOC are the same set
# --------------------------------------------------------------------------
head_ 6 "root parameters and $PARAMS_DOC are the same set"
if [ -z "$PY_YAML" ] || [ ! -f "$ROOT_T" ]; then
  skip 6 "needs PyYAML and $ROOT_T"
elif [ ! -f "$PARAMS_DOC" ]; then
  fail 6 "$PARAMS_DOC: expected to exist"
else
  cfnq params "$ROOT_T" | sort -u > "$TMP/root-params.txt" || true
  : > "$TMP/all-params.txt"
  for t in "${ALL_T[@]}"; do
    [ -f "$t" ] || continue
    cfnq params "$t" >> "$TMP/all-params.txt" 2>/dev/null || true
  done
  sort -u -o "$TMP/all-params.txt" "$TMP/all-params.txt"
  # A documented parameter is the first cell of a table row, in backticks —
  # the shape aws-pcs/docs/PARAMETERS.md uses.
  grep -oE '^\|[[:space:]]*`[A-Za-z][A-Za-z0-9]*`' "$PARAMS_DOC" \
    | tr -d '|` ' | sort -u > "$TMP/doc-params.txt"
  if [ ! -s "$TMP/doc-params.txt" ]; then
    fail 6 "$PARAMS_DOC: no parameter rows found; expected Markdown table rows whose first cell is a backticked parameter name, as in architectures/aws-pcs/docs/PARAMETERS.md"
  fi
  while read -r prm; do
    [ -n "$prm" ] || continue
    grep -qx "$prm" "$TMP/doc-params.txt" || fail 6 "$ROOT_T declares parameter '$prm' but $PARAMS_DOC has no row for it"
  done < "$TMP/root-params.txt"
  while read -r prm; do
    [ -n "$prm" ] || continue
    grep -qx "$prm" "$TMP/all-params.txt" || fail 6 "$PARAMS_DOC documents '$prm', which no template declares (renamed or removed?)"
  done < "$TMP/doc-params.txt"
  pass 6 "$(wc -l < "$TMP/root-params.txt" | tr -d ' ') root parameters, $(wc -l < "$TMP/doc-params.txt" | tr -d ' ') documented"
fi

# --------------------------------------------------------------------------
# check 7 — relative links and in-page anchors resolve
# --------------------------------------------------------------------------
head_ 7 "relative links and anchors in README.md, docs/*.md and tests/*.md resolve"
if [ -z "$PY_ANY" ]; then
  skip 7 "no python3 on PATH"
else
  MD_FILES=()
  if [ -f README.md ]; then MD_FILES+=(README.md); fi
  for f in docs/*.md tests/*.md; do
    if [ -f "$f" ]; then MD_FILES+=("$f"); fi
  done
  if [ ${#MD_FILES[@]} -eq 0 ]; then
    fail 7 "no Markdown files found under README.md, docs/ or tests/"
  else
    if "$PY_ANY" "$TEXTQ" links "${MD_FILES[@]}" > "$TMP/links.txt" 2>&1; then
      pass 7 "${#MD_FILES[@]} file(s), every relative link and anchor resolves"
    else
      while read -r line; do
        [ -n "$line" ] || continue
        fail 7 "$line"
      done < "$TMP/links.txt"
    fi
  fi
fi

# --------------------------------------------------------------------------
# check 8 — nothing unpinned in the templates
# --------------------------------------------------------------------------
head_ 8 "no unpinned chart, image or download reference in the templates"
if [ -z "$PY_ANY" ]; then
  skip 8 "no python3 on PATH"
else
  T_PRESENT=()
  for t in "${ALL_T[@]}"; do
    if [ -f "$t" ]; then T_PRESENT+=("$t"); fi
  done
  if [ ${#T_PRESENT[@]} -eq 0 ]; then
    skip 8 "no templates present"
  elif "$PY_ANY" "$TEXTQ" pins "${T_PRESENT[@]}" > "$TMP/pins.txt" 2>&1; then
    pass 8 "${#T_PRESENT[@]} template(s), every chart, image and download pinned"
  else
    while read -r line; do
      [ -n "$line" ] || continue
      fail 8 "$line"
    done < "$TMP/pins.txt"
  fi
fi

# --------------------------------------------------------------------------
# check 9 — every output name the interface freezes exists in its template
# --------------------------------------------------------------------------
head_ 9 "the frozen output names exist in the template that owns them"
OUT_PREREQ="VpcId PublicSubnetId PrivateSubnetId ControlPlaneSubnetId NodeSecurityGroupId PrivateRouteTableId FsxFileSystemId FsxDnsName FsxMountName"
OUT_CLUSTER="ClusterName ClusterArn ClusterSecurityGroupId NodeRoleArn KubeconfigCommand"
OUT_GPU="GpuNodeGroupName GpuInstanceType GpuNodeCount BootstrapLogGroup BootstrapProjectName"
OUT_ROOT="ClusterName ClusterArn Region KubeconfigCommand VpcId PrivateSubnetId GpuNodeGroupName GpuInstanceType BootstrapLogGroup FsxFileSystemId FsxDnsName FsxMountName"
if [ -z "$PY_YAML" ]; then
  skip 9 "needs PyYAML"
else
  check_outputs() {   # $1 = template, $2 = space-separated expected names
    local t="$1" expected="$2" name
    if [ ! -f "$t" ]; then skip 9 "$t missing"; return 0; fi
    if ! cfnq outputs "$t" | sort -u > "$TMP/out.txt" 2>"$TMP/out.err"; then
      fail 9 "$t: could not read the Outputs section: $(tr '\n' ' ' < "$TMP/out.err")"
      return 0
    fi
    for name in $expected; do
      grep -qx "$name" "$TMP/out.txt" || fail 9 "$t: output '$name' is frozen in the interface document but the template does not declare it"
    done
    pass 9 "$t: $(wc -l < "$TMP/out.txt" | tr -d ' ') outputs, all $(printf '%s' "$expected" | wc -w | tr -d ' ') frozen names present"
  }
  check_outputs "$PREREQ_T" "$OUT_PREREQ"
  check_outputs "$CLUSTER_T" "$OUT_CLUSTER"
  check_outputs "$GPU_T" "$OUT_GPU"
  check_outputs "$ROOT_T" "$OUT_ROOT"
fi

# --------------------------------------------------------------------------
# check 10 — NicLayout agrees with the EC2 API
#
# SecondaryDeviceIndex is deliberately not checked here: describe-instance-types
# does not report which DeviceIndex the secondary cards must use, which is the
# reason that value is carried as data and verified on hardware instead.
# --------------------------------------------------------------------------
head_ 10 "NicLayout agrees with aws ec2 describe-instance-types"
if [ "$HAVE_AWS" != yes ]; then
  skip 10 "no usable AWS credentials (describe-instance-types is an authenticated call)"
elif [ ! -s "$TMP/NicLayout.tsv" ]; then
  skip 10 "NicLayout could not be read (see check 2)"
else
  REGION="$(aws configure get region 2>/dev/null || true)"
  if [ -n "$REGION" ]; then REGION=" in $REGION"; fi
  PRIMARY_UNVERIFIED=""
  while read -r t; do
    [ -n "$t" ] || continue
    if ! api=$(aws ec2 describe-instance-types --instance-types "$t" \
        --query 'InstanceTypes[0].[NetworkInfo.MaximumNetworkCards,NetworkInfo.EfaSupported,NetworkInfo.EfaInfo.MaximumEfaInterfaces]' \
        --output text 2>"$TMP/dit.err"); then
      skip 10 "$t: not offered${REGION} or not described: $(tr '\n' ' ' < "$TMP/dit.err" | cut -c1-140)"
      continue
    fi
    api_cards=$(printf '%s' "$api" | cut -f1)
    api_efa=$(printf '%s' "$api" | cut -f2)
    api_ifaces=$(printf '%s' "$api" | cut -f3)
    if [ "$api_ifaces" = None ]; then api_ifaces=0; fi
    map_cards=$(awk -F'\t' -v t="$t" '$1==t && $2=="Cards" {print $3; exit}' "$TMP/NicLayout.tsv")
    map_ifaces=$(awk -F'\t' -v t="$t" '$1==t && $2=="EfaInterfaces" {print $3; exit}' "$TMP/NicLayout.tsv")
    map_primary=$(awk -F'\t' -v t="$t" '$1==t && $2=="PrimaryEfa" {print $3; exit}' "$TMP/NicLayout.tsv")
    [ "$map_cards" = "$api_cards" ] || fail 10 "$t: NicLayout Cards=$map_cards but MaximumNetworkCards=$api_cards"
    [ "$map_ifaces" = "$api_ifaces" ] || fail 10 "$t: NicLayout EfaInterfaces=$map_ifaces but MaximumEfaInterfaces=$api_ifaces (EfaSupported=$api_efa)"
    card0=$(aws ec2 describe-instance-types --instance-types "$t" \
        --query 'InstanceTypes[0].NetworkInfo.NetworkCards[?NetworkCardIndex==`0`].EfaSupported' \
        --output text 2>/dev/null || true)
    case "$card0" in
      True|true)  [ "$map_primary" = true ]  || fail 10 "$t: NicLayout PrimaryEfa=$map_primary but network card 0 reports EfaSupported=true" ;;
      False|false) [ "$map_primary" = false ] || fail 10 "$t: NicLayout PrimaryEfa=$map_primary but network card 0 reports EfaSupported=false" ;;
      *) PRIMARY_UNVERIFIED="$PRIMARY_UNVERIFIED $t" ;;
    esac
    pass 10 "$t: Cards=$map_cards, EfaInterfaces=$map_ifaces"
  done < <(cut -f1 "$TMP/NicLayout.tsv" | sort -u)
  if [ -n "$PRIMARY_UNVERIFIED" ]; then
    skip 10 "PrimaryEfa is unverified for:$PRIMARY_UNVERIFIED — describe-instance-types does not return EfaSupported per network card, so this key and SecondaryDeviceIndex are both outside what the EC2 API can confirm; they are proven by tests/gpu-efa-test.md on hardware"
  fi
fi

# --------------------------------------------------------------------------
printf '\n'
if [ "$FAILURES" -eq 0 ]; then
  printf '[OK] lint-templates.sh: 0 failures, %d skipped\n' "$SKIPS"
  printf '     A skip is not a pass. Re-run with AWS credentials and helm on PATH to close the skips.\n'
  exit 0
fi
printf '[FAILED] lint-templates.sh: %d failure(s), %d skipped\n' "$FAILURES" "$SKIPS"
printf '         Every FAIL line above names the file, what was expected and what was found.\n'
exit 1
