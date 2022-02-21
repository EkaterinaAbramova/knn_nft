# Rust Smart Contract: K Nearest Neighbours Machine Learning Algo

## Algo Description

Contract uses K Nearest Neighbours ([KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)) algorithm to classify multidimensional datapoints into either class 0 (absence of something) or class 1 (presence of something). A client (e.g. a biological laboratory) can interact with this NFT in order to obtain the class (0: no cancer, or 1: cancer) of a patient's biopsy (i.e. a newly observed data point). 

Decision making of the KNN algorithm is based on calculating the Euclidean distance between the test point (one to be classified those class is unknown, e.g. cancer or not) and all of the points in the train dataset (those classes are known and available to the algorithm). 
Note that the client does not need to provide the full training dataset (the NFT contract has access to it - in the current version only a toy such dataset is programmed). Hence the client only needs to select what dataset they want to use and to provide the test data point with the required features (in the present toy data set example just 2 features are used, i.e. 2 columns of a dataset are present). 
The class of the test data point is assigned to be the majority class of its K nearest neighbours (i.e. K nearest training data points to the test point).

The contract allows a selection of datasets to the client, in the current version it provides a choice of two toy dataset examples: `TOY_CANCER_TRAIN` and `TOY_CUSTOMER_TRAIN`, both 10x2 arrays (i.e. 2 features) and their corresponding known target classes `TOY_CANCER_TARGET` and `TOY_CUSTOMER_TARGET` (10x1 arrays, known classes for each train data point). 

KNN algorithm's requirements:
- Algorithm takes one hyperparameter, `k`, which is a positive odd number typically in the range 1 and 15.
- `k` specifies how many nearest neighbours (i.e. closest to the test data point) the algorithm will look to in order to decide test data point's class.
- The full train dataset is needed each time a new test point classification is to be made. This means KNN is unlike other algorithms where learnt parameters is all that is necessary, making requirement for run time data access unnecessary.

KNN algorithm's steps:
- Calculates Euclidean distance between the test point and all of the points in the train dataset
- Orders the distances in ascending order
- Selects points corresponding to K first distances
- Examines the most frequent class observed in the selected K points
- Classifies test data point to majority class. 
See ([KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)) for a visual. 

## Prerequisites
  * Make sure Rust is installed as per the prerequisites in [`near-sdk-rs`](https://github.com/near/near-sdk-rs).
  * Make sure [near-cli](https://github.com/near/near-cli) is installed.

## Explore Contract
The source for this contract is in `knn_nft/src/lib.rs`. The contract allows a customer to select a toy dataset they want to work with _cancer_ or _customer_ and accepts input consisting of new unseen test data point, and as a result provides the class (class 0 or class 1) for this test point.

## Build Contract
Since the smart contract is a library (crate) rather than a binary, the `cargo run` command is unavailable. Instead to build the contract ensure your cmd is in the `knn_nft` path and use:
```bash
./build.sh
```
If you run into 'permission denied' difficulties, simply execute this line first:
```bash
chmod +x build.sh
```

## Test Contract
There are 6 unit tests in this contract, specifically: `test_default_k`, `test_new_k`, `test_run_analysis`, `test_calc_euclidean_dist`, `test_sort_and_argsort`, `test_classify_test_point` which are designed to verify that the contract code is working as indtended.
```bash
cargo test -- --nocapture
```

## Interact With The Contract

### Create Necessary Accounts

This smart contract is intended to be interacted with using a development account. Go to [NEAR URL](https://wallet.testnet.near.org) and create a testnet account, for example called `myacc.testnet`. For help see [Dev Accounts](https://docs.near.org/docs/concepts/account#dev-accounts). Following this run the command below, which lets the near cli generate a private key, kept in the jason file on your computer, and public key as a URL parameter to NEAR wallet by logging into your account from your Terminal (browswer opens up):
```bash
near login
```

Next, using the Terminal create a sub-account (for example `knn_nft.myacc.testnet`) to which the contract will be deployed to (this is the best practice for deploying contracts), using command:
```bash
near create-account knn_nft.myacc.testnet --masterAccount myacc.testnet
```

### Deploy The Contract to Blockchain
Ensure the cmd is in the dirctory containing `res` folder and run the following command, which will deploy theh contract to blockchain and at the same time initialise contract parameter `k` to the value of choice (in this case 3), using Batch Action:
```bash
near deploy knn_nft.myacc.testnet --wasmFile res/knn_supervised_learning.wasm --initFunction 'new' --initArgs '{"k": 3}'
```

### Obtain Test Point Class
Next, specify the toy dataset you would like to work with (`cancer` or `customer`) and provide a test data point (one those class is to be established). For example cancer data set with a test point [15.8, 2.0]:
```bash
near call knn_nft.myacc.testnet run_analysis '{"data_set": "cancer", "test_point": [13.9, 1.9]}' --accountId myacc.testnet
```
The established class is provided as a result.

**Get more info at:**

* [Rust Smart Contract Quick Start](https://docs.near.org/docs/develop/contracts/rust/intro)
* [Rust SDK Book](https://www.near-sdk.io/)
