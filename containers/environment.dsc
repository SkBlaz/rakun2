bootstrap: docker
From: ubuntu:latest

%labels

%environment
export LC_ALL=C

%files
    ../rakun2 /opt/rakun2.0/rakun2
	../MANIFEST.in /opt/rakun2.0/MANIFEST.in
	../requirements.txt /opt/rakun2.0/requirements.txt
	../setup.py /opt/rakun2.0/setup.py

%post
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y libhdf5-dev graphviz locales python3-dev python3-pip curl git zip
apt-get -y install nordugrid-arc-client nordugrid-arc-plugins-needed nordugrid-arc-plugins-globus
apt-get clean
cd /opt/rakun2.0; ls; pip install . --upgrade

unset DEBIAN_FRONTEND