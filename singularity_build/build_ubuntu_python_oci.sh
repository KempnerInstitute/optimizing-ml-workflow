XDG_RUNTIME_DIR=$(mktemp -d) singularity build --fakeroot --oci ubuntu_python.oci.sif dockerfile_ubuntu_python 

