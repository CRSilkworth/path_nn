# Path Neural Network
The code used in the analysis of infinite (or just very large) width and depth neural networks of the form:
![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26%20%5Csigma%5ED%20%5Ccirc%20g%5ED%20%5Ccirc%20%28%5Csigma%5E%7BD%7D%29%5E%7B-1%7D%20%5Ccirc%20.%20.%20.%20%5Ccirc%20%5Csigma%5E1%20%5Ccirc%20g%5E1%20%5Ccirc%20%28%5Csigma%5E%7B1%7D%29%5E%7B-1%7D%20%5C%5C%20%26%20%5C%5C%20%26%20with%3A%20%5C%5C%20%26%20g%5Ed_i%28x_j%29%20%3D%20%5Calpha%5Ed_i%20&plus;%20%5Csum_%7Bj%7D%5Cbeta%5Ed_%7Bij%7D%20x_j%2C%20%5C%20%5C%20%5Calpha%5Ed_i%20%5Cin%20%5Cmathbb%7BR%7D%5EW%2C%5C%20%5Cbeta%5Ed_%7Bij%7D%20%5Cin%20%5Cmathbb%7BR%7D%5EW%20%5Ctimes%20%5Cmathbb%7BR%7D%5EW%5C%5C%20%26%20%5Csigma%5Ed_i%28x_i%29%20%5Crightarrow%20invertible%2C%5C%20pointwise%20%5C%5C%20%5Cend%7Balign*%7D)

For example:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%26%20%5Csigma%5Ed_i%28x_i%29%20%3D%20x_i%5E%7Bn%28d%29%20&plus;%201%7D%20%5C%5C%20%26%20n%3A%20%5Cmathbb%7BR%7D%20%5Crightarrow%20%5B0%2C%20&plus;%5Cinfty%29%20%5Cend%7Balign*%7D)

This repo will look at various forms of n(d), the limit that D goes to infinity as well as the limit that W goes to infinity. As of this writing, analysis is done using mnist data but more will be added in the future.

## Set up

### Prerequisites
* Update system python 3
```
# MAC OS
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python

# Ubuntu
sudo apt-get install python3.6
sudo apt-get install \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    python3-pip git software-properties-common
sudo apt-get install python3.6-dev
```

* docker must be installed( [mac](https://docs.docker.com/docker-for-mac/install/), [ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/))


### Inference only install
* Pull tensorflow serving docker image
```
docker pull tensorflow/serving:2.0.0
```

### Install: Beam
* Checkout the mlp repository and build the mlp/beam docker imagine.
```
git clone --branch v{version} git+ssh://git@github.com/CRSilkworth/mlp.git
cd mlp
build . -f Dockerfile.beam -t mlp/beam:latest
cd {path_to}/path_nn
build . -f Dockerfile.beam -t path_nn/beam:latest
```

## Running a pipeline
### Using beam
* Ensure that the environmental variables from the installation section are set and you have activated the virtualenv.
* cd to some subproject directory (e.g. intent_classifier/polyglot):
```
cd path_nn/<mlp_subproject>/
```
* Create or edit a beam pipeline file from the pipelines directory ({mlp_subproject}/pipelines/beam/). Adjust any of the input variables, i.e. the variables uppercased beginning with an underscore (e.g. \_NUM_TRAIN_STEPS) and run:
```
docker run \
  --gpus all -it -v $PWD:/tmp -w /tmp --rm \
  path_nn/beam:latest \
  python {mlp_subproject}/pipelines/beam/{pipeline_file}
```
Outputs will be written to a directory path_nn/runs/ by default.
