# Tensorflow Serving 安装

## 安装bazel
```
chmod +x bazel-0.15.2-installer-linux-x86_64.sh
./bazel-0.15.2-installer-linux-x86_64.sh --user
```

## 设置环境变量
```
# ~/.bashrc
export PATH="$PATH:$HOME/bin"
```

## 安装必要的包
```
sudo pip install grpcio
sudo pip install mock
sudo pip install enum34
sudo pip install tensorflow-serving-api
sudo pip install tensorflow
```

## Clone TensorFlow Serving repo:
```
git clone https://github.com/tensorflow/serving
cd serving
```

## 开始编译
```
bazel build -c opt tensorflow_serving/...
```

## 测试是否编译成功
```
bazel test -c opt tensorflo
```

## 从本地编译ModelServer
```
bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
```

## 导出minist例子中的模型
```
bazel build -c opt //tensorflow_serving/example:mnist_saved_model
bazel-bin/tensorflow_serving/example/mnist_saved_model /tmp/mnist_model
```
或者, 如果安装了tensorflow-serving-api
```
python tensorflow_serving/example/mnist_saved_model.py /tmp/mnist_model
```
## 启动model server
```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/
```

## 测试client
```
python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=localhost:9000
```

## 可能遇到的问题
#### 1. protoc 版本不一致。

找到所使用的protoc版本:
```
$HOME/.cache/bazel/_bazel_joshualeung]$ find . -name protoc
```
查看protobuf版本
```
$./c3caffc75bc56baa5c3c705da4d3fbd9/execroot/tf_serving/bazel-out/host/bin/external/protobuf_archive/protoc --version
libprotoc 3.6.0
```
使本地的protobuf版本与该版本一致。
```
pip uninstall protobuf
pip install protobuf==3.6.0
```
#### 2. 缺少相关的工具
```
yum install automake
yum install libtool
```

## 参考
1. https://www.tensorflow.org/serving/setup
2. https://www.tensorflow.org/serving/serving_basic