use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize}; // imports involving serialization are used to bundle the code/storage so that it's ready for the blockchain.
use near_sdk::{env, near_bindgen};


// ------------------------------------ VARIABLES OUTSIDE OF CONTRACT (NO STAKING) -----------------------------
// Outside struct therefore won't be on the blockchain and thus won't require staking of NEAR tokens from developer's account
// Toy data for cancer dataset, and for customer data set. Arrays with train data  10x2 and target classes 10x1 (i.e. class that data point belongs to)
const TOY_CANCER_TRAIN: &'static [[f64; 2]; 10] = &[[1.4, 14.2], [7.3, 3.6], [15.8, 2.0], [7.0, 9.1], [13.9, 5.7], [16.6, 2.1], [18.1, 4.5], [8.1, 11.1], [11.9, 1.9], [12.8, 15.7]];
const TOY_CANCER_TARGET: &'static [u8] = &[0, 1, 1, 1, 0, 0, 1, 0, 1, 0];
const TOY_CUSTOMER_TRAIN: &'static [[f64; 2]; 10] = &[[11.4, 4.2], [17.3, 13.6], [5.8, 22.0], [7.0, 1.1], [13.9, 5.7], [16.6, 9.1], [8.1, 1.5], [1.1, 11.1], [2.9, 19.9], [22.8, 15.7]];
const TOY_CUSTOMER_TARGET: &'static [u8] = &[1, 0, 0, 1, 1, 0, 1, 1, 1, 0];

//When writing smart contracts, the pattern is to have a struct with an associated impl where you write the core logic into functions.
// ------------------------------------------ CONTRACT STATE --------------------------------------------------
#[near_bindgen] // macro: allow the compilation into WebAssembly to be compatible and optimized for the NEAR blockchain.
#[derive(BorshDeserialize, BorshSerialize)] // deleted Default since have 'default constructor' below
pub struct KnnMachineLearning { // name of my Contract K Nearest Neighbours Classification Algorithm
    param_k: u8, // number of nearest neighbours (MUST BE odd value between 1 and 15) 
    // u8 is suitable since it takes unsigned values (0,255) and k has at the lowest value 1, and at the highest approx 15.
}

// ------------------------------------------ CONTRACT METHODS -------------------------------------------------
// 'Default constructor'. Allows to instantiate the struct by giving only the non-default values: let p = KnnMachineLearning {var: 10, ..Default::default()};
impl Default for KnnMachineLearning {
    fn default() -> KnnMachineLearning {
        KnnMachineLearning {
            param_k: 5, // typical k value is 3, 5, or 7. Therefore making a default popular choice of k=5.
            // Here staking will be required as the information is stored on the blockchain.
        }
    }
}

// Functions I will be invoking on the blockchain:
#[near_bindgen] // macro: allow the compilation into WebAssembly to be compatible and optimized for the NEAR blockchain.
impl KnnMachineLearning {
    #[init]
    // This is a public method which is exported to the contract i.e. anyone can call it. 
    pub fn new(k: u8) -> Self { // could set another k value during depolyment using Batch Action. 
        assert_eq!((k % 2 != 0) & (k > 0) & (k <= 15), true, "k must be positive and odd between 1 and 35!"); // Algo requirement: ensure k is positive odd number between 1 and 15
        Self {
            param_k : k,
        }
    }

    // near_sdk: method is VIEW if &self; method is CHANGE if &mut self.
    // CHANGE methods serialize the main contract structure at the end and store the new value into storage.
    // Made this mutable to allow change of state in the contract. (Data scope should ensure it is destroyed and thus (hopefully) not stored into staked memory)
    pub fn run_analysis(&mut self, data_set: &String, test_point: &[f64; 2]) -> u8 { // dataset has 2 columns/features, hence test point needs to have same dimensionality. 
        // Dataset can either be 'cancer' or 'customer as provided by the user.
        let mut ans: u8 = 0;
        if data_set == "cancer" {
            env::log_str("Working with cancer dataset.");
            // call fn to do the calculations with CANCER toy data
            ans = self.classify_test_point(&TOY_CANCER_TRAIN, &TOY_CANCER_TARGET, &test_point); // borrow data and test point to fn classify_test_point
        } else if data_set == "customer" {
            env::log_str("Working with customer dataset.");
            // call fn to do the calculations with CUSTOMER toy data
            ans = self.classify_test_point(&TOY_CUSTOMER_TRAIN, &TOY_CUSTOMER_TARGET, &test_point);
        } else {
            env::log_str("Data can either be: 'cancer' or 'customer' data. Re-specify.");
        };
        println!("The test point class is: {}", ans);
        ans
    }
    
    // Fn callable from inside contract methods only, not by user. Parameters: array 10x2, array 10x1, array 2x1.
    fn classify_test_point(&self, arr_train: &[[f64; 2]; 10], arr_target: &[u8], pt: &[f64; 2]) -> u8 {
        // Get L2 norm (Euclidean) distances from test point to all train data points
        let dist = self.calc_euclidean_dist(&arr_train, &pt);
        // Sort distances in ascending order. Obtain argsort() of that action and re-order corresponding target labels (keep train point distances and target classes aligned).
        let (indices, _sorted_distances) = self.sort_and_argsort(&dist);
        // Based on indices obtained from argsort() re-order targets
        let sorted_targets = indices.into_iter().map(|x| arr_target[x]).collect::<Vec<u8>>();
        // Obtain the classes of k nearest neighbours (distances were sorted in ascending order, so take first k elements from sorted_targets)
        let first_k: Vec<u8> = sorted_targets[0..(self.param_k as usize)].to_vec();
        // Count number of classes with label 1 vs label 0, go with majority
        let n_1: usize = first_k.iter().filter(|&n| *n == 1).count(); // # of 1s 
        let n_0: usize = first_k.iter().filter(|&n| *n == 0).count(); // # of 0s
        if n_1 > n_0 {
            1
        } else {
            0
        }
    }

    // Callable from methods only (not user). Params: array 10x2, array 2x1.
    fn calc_euclidean_dist(&self, arr_train: &[[f64; 2]; 10], pt: &[f64; 2]) -> Vec<f64> {
        let mut dist: Vec<f64> = Vec::new(); // store distanes 
        for obs in arr_train { // for each observation in train dataset i.e. obs=[x, y]
            let mut sum_sq_diff: f64 = 0.0; // sum of squared differences 
            for ii in 0..obs.len(){ // go over each dim of the train point obs (note: sequence stop index is decremented by 1 automatically therefore 0 to len is correct)
                sum_sq_diff += (obs[ii] - pt[ii]).powi(2);  // square the diff and add 
            }
            // Once the sum is completed obtain the Euclidean distance
            dist.push(sum_sq_diff.sqrt());
        }
        dist
    }

    // Callable from methods only (not user). Parameters: vec 10x1.
    fn sort_and_argsort(&self, vec: &Vec<f64>) -> (Vec<usize>, Vec<f64>) {
        let v_original = vec.clone(); // avoid handing over owenership
        let mut v = vec.clone(); // avoid handing over owenership
        // sort v in-place
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // implement argsort() equivalent
        let mut inds = Vec::new();
        for ii in 0..v.len() {
            let ans = v_original.iter().position(|&r| r == v[ii]).unwrap();  
            inds.push(ans);
        }
        (inds, v) // return 2 variables
    }
}


// ---------------------------------------------- TESTS ----------------------------------------------------------
#[cfg(test)]
mod tests { // start of unit tests
    use super::*;
    use near_sdk::test_utils::{get_logs, VMContextBuilder};
    use near_sdk::{testing_env, AccountId};

    // Set up a mock context. Provide a `predecessor` here, it'll modify the default context.
    fn get_context(predecessor: AccountId) -> VMContextBuilder {
        let mut builder = VMContextBuilder::new();
        builder.predecessor_account_id(predecessor);
        builder
    }

    // TESTS HERE
    #[test] 
    fn test_default_k() { // Check that the default k value is 5
        let contract = KnnMachineLearning::default(); 
        assert_eq!(contract.param_k == 5, true, "Expected default value for k=3") 
    }
    
    #[test]
    fn test_new_k() { // Check that initialisation of k upon deployment satisfies requirements of being +ve, odd number between 1 and 15
        KnnMachineLearning::new(3); // assert present inside new code
    }

    #[test]
    fn test_run_analysis() { // run_analysis is the top level method. Here will test that datset name was correctly specified
        let mut contract = KnnMachineLearning::new(3);
        let test_point: [f64; 2] = [2.2, 14.0]; // vector with 2 entries
        contract.run_analysis(&"cancer".to_string(), &test_point);
        contract.run_analysis(&"customer".to_string(), &test_point);
        contract.run_analysis(&"wrong dataset".to_string(), &test_point);
        assert_eq!( //Asserts that two expressions are equal to each other 
            get_logs(), 
            ["Working with cancer dataset.", "Working with customer dataset.", "Data can either be: 'cancer' or 'customer' data. Re-specify."],
            "Expected a successful log."
        );
    }

    #[test]
    fn test_calc_euclidean_dist() { // check knn algo's sub-tasks work correctly
        let contract = KnnMachineLearning::new(3);
        let test_point: [f64; 2] = [15.8, 2.0]; // vector with 2 entries
        let d = contract.calc_euclidean_dist(&TOY_CANCER_TRAIN, &test_point);
        let mut rounded_d = Vec::new();
        for elem in d {
            rounded_d.push((elem * 100.0).round() / 100.0);
        }
        assert_eq!(rounded_d, vec![18.87, 8.65, 0.00, 11.31, 4.16, 0.81, 3.40, 11.92, 3.90, 14.02], "Expected equality."); // Correct answer obtained from the correct code in Python.
    }

    #[test]
    fn test_sort_and_argsort() { // check knn algo's sub-tasks work correctly
        let contract = KnnMachineLearning::new(3);
        let v = vec![1.1, 7.1, 4.1, 2.1]; // vector of floats
        let (i, v_sorted) = contract.sort_and_argsort(&v);
        assert_eq!(i, vec![0, 3, 2, 1], "Expected equality."); //Asserts that two expressions are equal to each other 
        assert_eq!(v_sorted, vec![1.1, 2.1, 4.1, 7.1], "Expected equality."); // Correct answer can be obtained by visual inspection.
    }

    #[test]
    fn test_classify_test_point(){ // check single test data point and 10 test data points for class results.
        let contract = KnnMachineLearning::new(3);
        // Test a single data point
        let test_point: [f64; 2] = [13.9, 1.9]; // vector with 2 entries
        let ans = contract.classify_test_point(&TOY_CANCER_TRAIN, &TOY_CANCER_TARGET, &test_point);
        assert_eq!(ans, 1, "Expected equality."); // This data point should be classified as 1, established from Python code.
        // Test 10 data points: the data points from the training set (note: they will not ALL be classified correctly as algo has some error; expected result given below as tested in Python)
        let test_points = TOY_CANCER_TRAIN.clone(); // array with 10 entries   
        let mut pred_class = vec![0; (test_points.len() as u8).into()]; // store predicted class labels.
        let mut count = 0;
        for pt in test_points { // go over test points (note each is 2x1)
            let ans = contract.classify_test_point(&TOY_CANCER_TRAIN, &TOY_CANCER_TARGET, &pt);
            pred_class[count] = ans; // store predicted class one at a time (for each test point)
            count += 1;
        }
        assert_eq!(pred_class, vec![0, 1, 1, 1, 1, 1, 1, 0, 1, 0], "Expected equality."); // correct classes (obtained with code in Python)
    }
}

// ------------------------------------------------- NOTES FOR ME -------------------------------------------------------
/* 
DEPLOYMENT NOTES
If you do changes to the contract, re-build, delete sub-account, then re-deploy. 
- In Terminal, run:
      $ near login 
     near cli generated private key (kept in jason file on computer) and public key as a URL param to NEAR wallet -> browser opens up, log into the testnet account.
1. Build contract and run all tests (ensure all are passed)
    $ ./build.sh
    $ cargo test -- --nocapture
2. Create sub-account (or delete and re-create it)
   This will clear the state and give a fresh start (also delete will transfer back the 100 NEAR tokens back into parent account):
    $ near delete knn_nft.drkat.testnet drkat.testnet  
    $ near create-account knn_nft.drkat.testnet --masterAccount drkat.testnet
   
   Can view subaccount state:
    $ near state knn_nft.drkat.testnet
   Account knn_nft.drkat2.testnet:
    {
    amount: '100000000000000000000000000',
    block_hash: '7NLf8towtjiBKW9T3puz462yb113yXbBM6zf9FP8rAAu',
    block_height: 83353892,
    code_hash: '11111111111111111111111111111111',  // i.e. no contract deployed
    locked: '0',
    storage_paid_at: 0,
    storage_usage: 182,
    formattedAmount: '100'
    }
3. Deploy to sub-account and initialise state
   Ensure the cmd is in the dirctory containing res folder.
    $ near deploy knn_nft.drkat.testnet --wasmFile res/knn_supervised_learning.wasm
   Contract is deployed, next can call the new init method with specific k value. 
    $ near call knn_nft.drkat.testnet new '{"k": 3}' --accountId knn_nft.drkat.testnet

   A safer approach is to use Batch Action (to ensure initialisation happens together with deployment) using specific value for k:
    $ near deploy knn_nft.drkat.testnet --wasmFile res/knn_supervised_learning.wasm --initFunction 'new' --initArgs '{"k": 3}'
    
   See the transaction in the transaction explorer https://explorer.testnet.near.org/transactions/9U7dNEg46p3LdJstkSFWdd86tQb8ogqGp6mZr6dYXB2A 
   View state again to see that the contract is now deployed (i.e. code_hash is not 1s):
    $ near state knn_nft.drkat.testnet
   Account knn_nft.drkat.testnet
    {
    amount: '99999304079890545800000000',
    block_hash: '3mmnSeD9MKNvWRTtNxXRCrA2SH3Wc3aztdQrr5iN86Lk',
    block_height: 83353994,
    code_hash: 'Hr3bkrHmwBv3FDjFn7K3GYmVeMUaz5u9vz6rekEv1fFQ', // now see that a contract is deployed
    locked: '0',
    storage_paid_at: 0,
    storage_usage: 105750,
    formattedAmount: '99.9993040798905458'
    }
4. Interact
   Specify the data set you want to work with (either "cancer" or "customer"); provide test point [13.9, 1.9] which is a 2x1 array, and obtain the class (for this example should be 1): 
    $ near call knn_nft.drkat.testnet run_analysis '{"data_set": "cancer", "test_point": [13.9, 1.9]}' --accountId drkat.testnet  
*/

/*
KNN BY HAND IN PYTHON:
n = len(X_test)
m = len(X_train)
y_pred = [0] * n # empty list
for ii in range(n):
    distVec = pd.Series(np.zeros(m)) # Pandas Series of size m (all 0s)
    a = X_test[ii, :] # row of test data (i.e. 1 test point)
    for jj in range(m):
        b = X_train[jj, :] # row of train data (1 train point)
        sqDiff = (a - b) ** 2
        distVec[jj] = sqrt(sqDiff.sum()) # distance to trainig point jj
    # Sort distance vector and corresponding training labels
    inds = distVec.argsort()
    distVec_sorted = distVec[inds]
    y_train_sorted = y_train[inds] # obtain y train in order of sorted dist vec
    # Obtain classification and compare to true answer
    firstK = y_train_sorted[0:K]
    unique, counts = np.unique(firstK, return_counts = True)
    y_pred[ii] = unique[counts.argmax()]
np.mean(y_pred == y_test) # 0.9473684210526315

KNN EQUIVALENT IN RUST:
// Fn callable from inside contract methods only, not by user. Parameters: array 10x2, array 10x1, array 2x1.
fn classify_test_point(&self, arr_train: &[[f64; 2]; 10], arr_target: &[u8], pt: &[f64; 2]) -> u8 {
    // Get L2 norm (Euclidean) distances from test point to all train data points
    let mut dist: Vec<f64> = Vec::new(); // store distanes 
    for obs in arr_train { // for each observation in train dataset i.e. [x, y]
        let mut sum_sq_diff: f64 = 0.0; // sum of squared differences between distances of individual dimensions of the 2 data points (the train point and test point)
        for ii in 0..obs.len(){ // go over each dimension of the train point given by obs (note: sequence stop is decremented by 1 automatically therefore 0 to len is correct)
            sum_sq_diff += (obs[ii] - pt[ii]).powi(2); 
        }
        // Once the sum is complete obtain the Euclidean distance
        dist.push(sum_sq_diff.sqrt());
    }
    // Once all distances are obtained sort dist in ascending order; obtain argsort() of that action and reorder corresponding target labels.
    let (indices, _sorted_distances) = self.sort_and_argsort(&dist);
    let sorted_targets = indices.into_iter().map(|x| arr_target[x]).collect::<Vec<u8>>();
    // Obtain k nearest neighbours (distances were sorted in ascending order, so take first k items)
    let first_k: Vec<u8> = sorted_targets[0..(self.param_k as usize)].to_vec();
    let n_1: usize = first_k.iter().filter(|&n| *n == 1).count(); // count number of points belonging to class 1
    let n_0: usize = first_k.iter().filter(|&n| *n == 0).count(); // count number of points belonging to class 0
    // output classification result (go with the majority class)
    if n_1 > n_0 {
        1
    } else {
        0
    }
}
*/

/*
RUST NOTES:
- Indent code shortcut: cmd + ] 
- In Rust by default everything (all variables) is PRIVATE!!! Need to use &mut to ensure can change values of variables.
- Rust is a statically typed.
- Indexing starts from 0.
- i32 is default integer.
- f64 is default float.
- '' char literals.
- "" string literals.
- Syntax 1_000 means integer 1000.
- Compiling in release mode won't check for integer overflow!
- Rust won't auto convert non-Boolean types to a Boolean for if statements. 
- Structs and enums are the building blocks for creating new types.
- Structs - custom data type that lets you name and package together multiple related values.
- Structs and enums have data
- #[expr] is an outer attribute (specifying attributes on an item)

Fundamental data types:
    scalar types: integers, floating-point numbers, Booleans (true/false), characters.
    primitive compound types: 
        tuples 
        arrays:  all elems same type; fixed length (# elems doesn't change); [1,2,3]. Allocated on stack.

Std Lib:
    vector: allowed to grow.

Expressions do not include ending semicolons.
{
    let x = 3;
    x + 1 // if put ; at the end here, will change expression to a statement. 
}
Statements don’t evaluate to a value.

Fns return the last EXPRESSION implicitly (no need for 'return').
We don’t name return values.
MUST declare return value's type after an arrow (->)
fn five() -> i32 {
    5
}
Funciton names follow snake convention by style guide my_funciton_name.
It is not typical to have getter methods (on structs) in Rust.
*/