

sudo apt-get update;

sudo apt-get install -y build-essential libssl-dev uuid-dev libgpgme11-dev \
     squashfs-tools libseccomp-dev wget pkg-config git cryptsetup debootstrap;


wget https://dl.google.com/go/go1.13.linux-amd64.tar.gz;

sudo tar --directory=/usr/local -xzvf go1.13.linux-amd64.tar.gz;

export PATH=/usr/local/go/bin:$PATH;

wget https://github.com/singularityware/singularity/releases/download/v3.5.3/singularity-3.5.3.tar.gz

tar -xzvf singularity-3.5.3.tar.gz


cd singularity;

./mconfig;

cd builddir;

make;

sudo make install;

. etc/bash_completion.d/singularity;

sudo cp etc/bash_completion.d/singularity /etc/bash_completion.d/;

singularity run library://godlovedc/funny/lolcow;
