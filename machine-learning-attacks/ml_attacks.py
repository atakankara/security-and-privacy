import sys
import random

import numpy as np
import pandas as pd
import copy

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_model(model_type):
    if model_type == "DT":
        return DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == "LR":
        return LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    elif model_type == "SVC":
        return SVC(C=0.5, kernel='poly', random_state=0)

def insert_trigger(data, labels, n):
    indices = np.where(labels==1)[0][0:n]

    for i in indices:
        data[i, 0] += 10
        data[i, 1] += 10
        data[i, 2] += 50
        data[i, 3] += 100

    return data


###############################################################################
############################### Label Flipping ################################
###############################################################################

def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    accuracy = []

    number_of_flipped_indices = int(n*y_train.shape[0])

    for i in range(100):
        flipped_training_data = copy.deepcopy(y_train)
        indices = np.random.choice(y_train.shape[0], number_of_flipped_indices, replace=False)
        flipped_training_data[indices] = abs(y_train[indices] - 1)

        model = get_model(model_type)
        model.fit(X_train, flipped_training_data)
        prediction = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, prediction))

    return np.mean(accuracy)
    

###############################################################################
################################## Backdoor ###################################
###############################################################################

def backdoor_attack(X_train, y_train, model_type, num_samples):    
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data

    x_generated_data = []
    num_generated_data = 20

    for i in range(num_generated_data):
        x_generated_data.append([np.random.uniform(low=min(X_train[:, 0]), high=max(X_train[:,0])),
                                np.random.uniform(low=min(X_train[:, 1]), high=max(X_train[:,1])),
                                np.random.uniform(low=min(X_train[:, 2]), high=max(X_train[:,2])),
                                np.random.uniform(low=min(X_train[:, 3]), high=max(X_train[:,3]))])

    x_generated_data = np.asarray(x_generated_data)
    y_generated_data = np.repeat(1, num_generated_data)
    x_generated_data = insert_trigger(x_generated_data, y_generated_data, num_generated_data)

    backdoor_inserted_train_data = copy.deepcopy(X_train)
    backdoor_inserted_train_data = insert_trigger(backdoor_inserted_train_data, y_train, num_samples)

    model = get_model(model_type)
    model.fit(backdoor_inserted_train_data, y_train)
    
    prediction = model.predict(x_generated_data)
    return accuracy_score(y_generated_data, prediction)

    

###############################################################################
############################## Evasion ########################################
###############################################################################

def evade_model(trained_model, actual_example):
    # TODO: You need to implement this function!
    actual_class = trained_model.predict([actual_example])[0]
    target_class = abs(actual_class - 1)
    modified_example = copy.deepcopy(actual_example)

    pred_class = actual_class

    while pred_class == actual_class:
        perturbation_of_strategies = [0]*8 #4 features and two ways to go (positive or negative)

        for i in range(actual_example.shape[0]):
            for j in [1, -1]:
                experimental_example = copy.deepcopy(modified_example)
                pred_class = trained_model.predict([experimental_example])[0]
                while pred_class == actual_class:
                    experimental_example[i] += j*0.5
                    pred_class = trained_model.predict([experimental_example])[0]
                    
                    if abs(experimental_example[i]) >100:
                        break

                if j == 1:
                    perturbation_of_strategies[2*i] = calc_perturbation(actual_example, experimental_example)
                else:
                    perturbation_of_strategies[2*i + 1] = calc_perturbation(actual_example, experimental_example)
        
        strategy = np.argmin(perturbation_of_strategies)
        index = int(strategy / 2)
        positive = strategy % 2 == 0

        if positive:
            modified_example[index] += 0.5
        else:
            modified_example[index] -= 0.5

        pred_class = trained_model.predict([modified_example])[0]

    return modified_example

def calc_perturbation(actual_example, adversarial_example):
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i]-adversarial_example[i])
        return tot/len(actual_example)

###############################################################################
############################## Transferability ################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    # TODO: You need to implement this function!
    crafted_by_DT = [evade_model(DTmodel, example) for example in actual_examples]
    crafted_by_LR = [evade_model(LRmodel, example) for example in actual_examples]
    crafted_by_SVC = [evade_model(SVCmodel, example) for example in actual_examples]

    SVC_truth = SVCmodel.predict(actual_examples)
    DT_truth = DTmodel.predict(actual_examples)
    LR_truth = LRmodel.predict(actual_examples)

    print("Crafted from DT and tested on SVC: ", np.sum(SVCmodel.predict(crafted_by_DT) != SVC_truth))
    print("Crafted from DT and tested on LR: ",  np.sum(LRmodel.predict(crafted_by_DT) != LR_truth))

    print("Crafted from SVC and tested on LR: ", np.sum(LRmodel.predict(crafted_by_SVC) != LR_truth))
    print("Crafted from SVC and tested on DT: ", np.sum(DTmodel.predict(crafted_by_SVC) != DT_truth))

    print("Crafted from LR and tested on DT: ", np.sum(DTmodel.predict(crafted_by_LR) != DT_truth))
    print("Crafted from LR and tested on SVC: ", np.sum(SVCmodel.predict(crafted_by_LR) != SVC_truth))



###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    # This function should return the STOLEN model, but currently it returns the remote model
    # You should change the return value once you have implemented your model stealing attack

    labels = remote_model.predict(examples)
    target_model = get_model(model_type)
    target_model.fit(examples, labels)
    return target_model
    

###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ## 
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##

def main():
    data_filename = "BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]
    
    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)    
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    
    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))
    
    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))
    
    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # # Label flipping attack executions:
    model_types = ["DT", "LR", "SVC"]
    n_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for n in n_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n)
            print("Accuracy of poisoned", model_type, str(n), ":", acc)
    
    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)
    
    # Evasion attack executions:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 50
    total_perturb = 0.0
    for trained_model in trained_models:
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
    print("Avg perturbation for evasion attack:", total_perturb/num_examples)
    
    # Transferability of evasion attacks:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 100
    evaluate_transferability(myDEC, myLR, mySVC, X_test[num_examples:num_examples*2])
    
    # Model stealing:
    budgets = [5, 10, 20, 30, 50, 100, 200]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))
    
    

if __name__ == "__main__":
    main()
