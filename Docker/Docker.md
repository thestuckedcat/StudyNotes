# 0. What is Docker

Docker是一种容器技术(A tool for creating and managing containers)

容器是一个标准的软件单元，也是代码与其依赖的集合。它保证了代码运行的一致性（不用配环境了，也不用理清多如牛毛的版本关系）



Docker拥有一个轻量级的操作系统以及一个Docker Engine，这使得它比虚拟机拥有更好的性能

![image-20240227134004380](./assets/image-20240227134004380.png)

![image-20240227134119375](./assets/image-20240227134119375.png)









# 1. Docker Setup

### 安装Docker

![image-20240227141547687](./assets/image-20240227141547687.png)

我们澄清一下，macOS和Windows本身不支持docker，因此它需要跑一个轻量级virtual os来安置docker engine,而Linux本身兼容docker engine，因此没必要。

总结一下，如果你的系统不是linux，你需要docker desktop为你安装docker engine。如果你的系统又太老，你需要docker toolbox帮助你安装docker desktop来安装docker engine。

额外的，docker hub能够允许我们在云，网络上托管映像，以方便的共享，这些是后话。

* 为Linux安装https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

  * Docker为Linux22.04即以上版本推出了其desktop，考虑到我的环境仍然为20.04，因此使用Docker Engine

  





### 使用VSCode作为编辑器

![image-20240227231325728](/home/stuckedcat-3060/Documents/Github_storage/StudyNotes/Docker/assets/image-20240227231325728.png)













## 2. Quick Start of Docker

Container总是基于Image的，因此首先需要创建一个Image

* 创建`Dockerfile`，这就是文件名，没有后缀，我们在Dockerfile中向Docker描述容器将如何设置

* 运行`docker build .`以创建这个Dockerfile对应的Image

  * 此时，docker会抓取当前的环境变量，并且从云端下载对应的copy
  * 然后，它会给我们一个Image（ID），这个Image已经准备好开始作为一个容器
  * 我们可以使用`docker run d2cc7b04fb0a`，其中那串字符就是给出的Image ID
    * 注意，默认情况下container与当前操作系统是独立的，如果你想要和当前操作系统产生连接，例如发送HTTP请求，你需要在Dockerfile中指定local port`EXPOSE 3000`，然后使用`docker run -p 3000:3000 d2cc7b04fb0a`，其中3000:3000代表publish port 3000 on port 3000，这使得我们可以使用本地系统的本地主机访问port3000上运行的应用程序，而非容器。
  * 如果你使用desktop，你需要保证在运行这一句时，desktop已经启动并且在后台

* 终止docker:

  * `docker ps`:列出所有正在运行的容器，你会发现docker自动为这个container分配了一个名称，例如`relaxed_bartik`
  * `docker stop relaxed_bartik`

  
