from nvcr.io/nvidia/deepstream-l4t:6.0-samples
run bash install.sh
run mkdir /app
workdir /app
run apt-get update && apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
   libgstrtspserver-1.0-dev libx11-dev