IMAGE=registry.gitlab.aws.dev/smml/benchmarking/tools/preflight
DEPLOY_IMAGE=deploy
AWS_REGION ?= us-east-1
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: ami ami_ec2_ubuntu2404 ami_ec2_ubuntu2404_dlami ami_pcluster_ubuntu2404 ami_eks_al2023 ami_eks_ubuntu2404 ami_pcs_ubuntu2404

container_build:
	docker build -t ${IMAGE} ./preflight
container_run:
	docker run --privileged --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/uverbs1 --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs3 --gpus=all ${IMAGE}
container: container_build container_run

ami_ec2_ubuntu2404:
	packer build -only 'ec2-ubuntu2404.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee ec2-ubuntu2404_ami.log
ami_ec2_ubuntu2404_dlami:
	packer build -only 'ec2-ubuntu2404-dlami.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee ec2-ubuntu2404-dlami_ami.log
ami_pcluster_ubuntu2404:
	packer build -only 'pcluster-ubuntu2404.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee pcluster-ubuntu2404_ami.log
# Every override below is optional and is only passed when set, so packer-ami.pkr.hcl stays the single
# place the defaults live. README.md lists them and what each one is for.
ami_eks_al2023:
	packer build -only 'eks-al2023.*' -var aws_region=${AWS_REGION} -var "ami_version=1" \
	  $(if $(EKS_VERSION),-var eks_version=$(EKS_VERSION),) \
	  $(if $(INSTANCE_TYPE),-var instance_type=$(INSTANCE_TYPE),) \
	  $(if $(NVIDIA_DRIVER_VERSION),-var nvidia_driver_package_version=$(NVIDIA_DRIVER_VERSION),) \
	  $(if $(NVIDIA_KERNEL_MODULES),-var nvidia_driver_kernel_modules=$(NVIDIA_KERNEL_MODULES),) \
	  $(if $(NVIDIA_CONTAINER_TOOLKIT_VERSION),-var nvidia_container_toolkit_version=$(NVIDIA_CONTAINER_TOOLKIT_VERSION),) \
	  $(if $(NVIDIA_VERIFY_GPU),-var nvidia_driver_verify_gpu=$(NVIDIA_VERIFY_GPU),) \
	  $(if $(PCLUSTER_AMI_REGION),-var pcluster_ami_region=$(PCLUSTER_AMI_REGION),) \
	  packer-ami.pkr.hcl | tee eks-al2023_ami.log
ami_eks_ubuntu2404:
	packer build -only 'eks-ubuntu2404.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee eks-ubuntu2404_ami.log
ami_pcs_ubuntu2404:
	packer build -only 'pcs-ubuntu2404.*' -var aws_region=${AWS_REGION} -var "ami_version=1" packer-ami.pkr.hcl | tee pcs-ubuntu2404_ami.log

ami:
	@echo "Select one AMI target explicitly; builds create billable AWS resources." >&2
	@echo "Targets: ami_ec2_ubuntu2404 ami_ec2_ubuntu2404_dlami ami_pcluster_ubuntu2404 ami_eks_al2023 ami_eks_ubuntu2404 ami_pcs_ubuntu2404" >&2
	@exit 2

deploy_build:
	docker build -t ${DEPLOY_IMAGE} ./test
deploy: deploy_build
	docker run -v ${HOME}/.aws:/root/.aws:ro -v ${shell pwd}/test:/tmp/test ${DEPLOY_IMAGE} pcluster create-cluster -n test-ami -r ${AWS_REGION} -c /tmp/test/cluster.yaml --rollback-on-failure false
