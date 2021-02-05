FROM ubuntu:latest

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y tmux python3 python3-pip htop git scala build-essential default-jdk aria2 openssh-server xrdp xfce4 xfce4-power-manager xfce4-terminal sudo && apt-get clean
RUN mkdir /var/run/sshd
RUN adduser --disabled-password --gecos '' vagrant
RUN adduser vagrant sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chown -R vagrant /home/vagrant
RUN echo 'vagrant:vagrant' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
USER vagrant
RUN echo "xfce4-session" > /home/vagrant/.xsession
USER root
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
USER vagrant
RUN cd /home/vagrant && aria2c -q "https://download.jetbrains.com/product?code=PC&latest&distribution=linux" && tar xf *.tar.gz && rm *.tar.gz
RUN cd /home/vagrant && git clone https://github.com/swerdnase1902/Current-Topics-in-Computer-System-Modeling-Analysis-Cloud-Computing.git
USER root
RUN cd /home/vagrant && aria2c -q "https://apache.osuosl.org/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz" && tar xf *.tgz && rm *.tgz && mv spark*/ /opt/spark
ENV SPARK_HOME "/opt/spark"
ENV PATH "$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin"
ENV PYSPARK_PYTHON "/usr/bin/python3"
RUN python3 -m pip install pyspark numpy scipy pandas sklearn boto boto3 awscli pillow matplotlib opencv-python python-resize-image
USER root
RUN cd /home/vagrant && aria2c -q "https://go.microsoft.com/fwlink/?LinkID=760868" && DEBIAN_FRONTEND="noninteractive" apt install ./*.deb && rm *.deb
RUN echo 'alias python=python3' >> /home/vagrant/.profile
RUN echo 'export SPARK_HOME=/opt/spark' >> /home/vagrant/.profile
RUN echo 'export PATH=$PATH:/opt/spark/bin:/opt/spark/sbin' >> /home/vagrant/.profile
RUN echo 'export PYSPARK_PYTHON=/usr/bin/python3' >> /home/vagrant/.profile
RUN echo 'alias python=python3' >> /home/vagrant/.bashrc
RUN echo 'export SPARK_HOME=/opt/spark' >> /home/vagrant/.bashrc
RUN echo 'export PATH=$PATH:/opt/spark/bin:/opt/spark/sbin' >> /home/vagrant/.bashrc
RUN echo 'export PYSPARK_PYTHON=/usr/bin/python3' >> /home/vagrant/.bashrc
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y firefox && apt-get clean
USER vagrant
RUN cd /home/vagrant && aria2c -q "https://download.jetbrains.com/product?code=IIU&latest&distribution=linux" && tar xf *.tar.gz && rm *.tar.gz
USER root
EXPOSE 22
EXPOSE 3389 4041 8080 34123 34124 34125 34126 34127 18080
CMD ["/usr/sbin/sshd", "-D"]
