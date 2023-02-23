import matplotlib.pyplot as plt
import numpy as np
import math
import random
import matplotlib

''' Globals '''
LABELS = ["", "frontpage", "news", "tech", "local", "opinion", "on-air", "misc", "weather", "msn-news", "health", "living", "business", "msn-sports", "sports", "summary", "bbs", "travel"]

""" 
    Helper functions
    (You can define your helper functions here.)
"""
def read_dataset(filename):
    """
        Reads the dataset with given filename.

        Args:
            filename (str): Path to the dataset file
        Returns:
            Dataset rows as a list of lists.
    """

    result = []
    with open(filename, "r") as f:
        for _ in range(7):
            next(f)
        for line in f:
            sequence = line.strip().split(" ")
            result.append([int(i) for i in sequence])
    return result


def get_counts(dataset):
    flattened_dataset = np.array([item for sublist in dataset for item in sublist])
    counts = [0] * (len(LABELS)-1)
    for record in flattened_dataset:
        counts[record - 1] += 1
    return counts


### HELPERS END ###


''' Functions to implement '''

# TODO: Implement this function!
def get_histogram(dataset: list):
    """
        Creates a histogram of given counts for each category and saves it to a file.

        Args:
            dataset (list of lists): The MSNBC dataset

        Returns:
            Ordered list of counts for each page category (frontpage, news, tech, ..., travel)
            Ex: [123, 383, 541, ..., 915]
    """

    heights = get_counts(dataset)

    plt.figure(figsize=(16, 10))
    bars = LABELS[1:]
    x_positions = np.arange(len(bars))
    
    # Create bars and choose color
    plt.bar(x_positions, heights)
    # Add title and axis names
    plt.title('# of visits in each category')
    plt.ylabel('Counts')
    
    # Create names on the x axis
    plt.xticks(x_positions, bars)
    
    # Show graph
    plt.savefig("np-histogram.png")

    return heights



# TODO: Implement this function!
def add_laplace_noise(real_answer: list, sensitivity: float, epsilon: float):
    """
        Adds laplace noise to query's real answers.

        Args:
            real_answer (list): Real answers of a query -> Ex: [92.85, 57.63, 12, ..., 15.40]
            sensitivity (float): Sensitivity
            epsilon (float): Privacy parameter
        Returns:
            Noisy answers as a list.
            Ex: [103.6, 61.8, 17.0, ..., 19.62]
    """

    b = sensitivity / epsilon
    result = []

    for q in real_answer:
        result.append(q + np.random.laplace(0, b))
    
    return result


# TODO: Implement this function!
def truncate(dataset: list, n: int):
    """
        Truncates dataset according to truncation parameter n.

        Args:  
            dataset: original dataset 
            n (int): truncation parameter
        Returns:
            truncated_dataset: truncated version of original dataset
    """
    result = []
    
    for user_record in dataset:
        
        if len(user_record) > n:
            result.append(user_record[0:n])
        else:
            result.append(user_record)
    
    return result


# TODO: Implement this function!
def get_dp_histogram(dataset: list, n: int, epsilon: float):
    """
        Truncates dataset with parameter n and calculates differentially private histogram.

        Args:
            dataset (list of lists): The MSNBC dataset
            n (int): Truncation parameter
            epsilon (float): Privacy parameter
        Returns:
            Differentially private histogram as a list
    """
    
    truncated_dataset = truncate(dataset, n)
    
    counts = get_histogram(truncated_dataset)
    
    noisy_counts = add_laplace_noise(counts, n, epsilon)
    
    return noisy_counts

# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    """
        Calculates error according to the equation stated in part (e).

        Args: Actual histogram (list), Noisy histogram (list)
        Returns: Error (Err) in the noisy histogram (float)
    """

    error = 0
    number_of_bins = len(actual_hist)
    
    for i in range(number_of_bins):
        error += abs(noisy_hist[i] - actual_hist[i])
    
    error = error / number_of_bins
    
    return error


# TODO: Implement this function!
def n_experiment(dataset, n_values: list, epsilon: float):
    """
        Function for the experiment explained in part (f).
        n_values is a list, such as: [1, 6, 11, 16 ...]
        Returns the errors as a list: [1256.6, 1653.5, ...] such that 1256.5 is the error when n=1,
        1653.5 is the error when n = 6, and so forth.
    """

    errors = []

    for n in n_values:
        total_error = 0
        actual_hist = get_histogram(dataset)
        truncated_dataset = truncate(dataset, n)
        counts = get_histogram(truncated_dataset)
        
        for _ in range(30):
            noisy_hist = add_laplace_noise(counts, n, epsilon) #Repetetive get_dp_histogram function call results with inefficient memory management. This way is more efficient.
            current_error = calculate_average_error(actual_hist, noisy_hist)
            total_error += current_error
        
        errors.append(total_error / 30)

    return errors


# TODO: Implement this function!
def epsilon_experiment(dataset, n: int, eps_values: list):
    """
        Function for the experiment explained in part (g).
        eps_values is a list, such as: [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
        Returns the errors as a list: [9786.5, 1234.5, ...] such that 9786.5 is the error when eps = 0.0001,
        1234.5 is the error when eps = 0.001, and so forth.
    """

    errors = []

    for epsilon in eps_values:
        total_error = 0
        actual_hist = get_histogram(dataset)
        truncated_dataset = truncate(dataset, n)
        counts = get_histogram(truncated_dataset)

        for _ in range(30):
            noisy_hist = add_laplace_noise(counts, n, epsilon) #Repetetive get_dp_histogram function call results with inefficient memory management. This way is more efficient.
            current_error = calculate_average_error(actual_hist, noisy_hist)
            total_error += current_error

        errors.append(total_error / 30)

    return errors

# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def extract(dataset):
    """
        Extracts the first 1000 sequences and truncates them to n=1
    """
    
    return truncate(dataset[0:1000], 1)


# TODO: Implement this function!
def most_visited_exponential(smaller_dataset, epsilon):
    """
        Using the Exponential mechanism, calculates private response for query: 
        "Which category (1-17) received the highest number of page visits?"

        Returns 1 for frontpage, 2 for news, 3 for tech, etc.
    """
    
    #q_f(D, category): # of visits in dataset D that is from this category.
    #S(q_F): 1 since only one record differs between original and neighbour datasets.

    sensitivity = 1
    counts = get_counts(smaller_dataset)

    numerators = [math.exp((epsilon*count) / (2*sensitivity)) for count in counts]
    denominator = sum(numerators)

    probabilities = [numerator / denominator for numerator in numerators]
   
    categoriese = list(range(1, len(LABELS)))
    return random.choices(categoriese, probabilities)[0]

# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list):
    """
        Function for the experiment explained in part (i).
        eps_values is a list such as: [0.001, 0.005, 0.01, 0.03, ..]
        Returns the list of accuracy results [0.51, 0.78, ...] where 0.51 is the accuracy when eps = 0.001,
        0.78 is the accuracy when eps = 0.005, and so forth.
    """

    counts = get_counts(dataset)
    ground_truth = np.argmax(counts) + 1

    results = []

    for epsilon in eps_values:
        
        accuracy = 0
        for _ in range(1000):
            return_value = most_visited_exponential(dataset, epsilon)
            
            if return_value == ground_truth:
                accuracy += 1
        
        results.append(accuracy / 1000)

    return results


# FUNCTIONS TO IMPLEMENT END #

def main():
    dataset_filename = "msnbc.dat"
    dataset = read_dataset(dataset_filename)
    get_histogram(dataset)

    print("**** N EXPERIMENT RESULTS (f of Part 2) ****")
    eps = 0.01
    n_values = []
    for i in range(1, 106, 5):
       n_values.append(i)
    errors = n_experiment(dataset, n_values, eps)
    for i in range(len(n_values)):
       print("n = ", n_values[i], " error = ", errors[i])   
    
    print("*" * 50)

    # print("**** EPSILON EXPERIMENT RESULTS (g of Part 2) ****")    
    # n = 50
    # eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    # errors = epsilon_experiment(dataset, n, eps_values)
    # for i in range(len(eps_values)):
    #    print("eps = ", eps_values[i], " error = ", errors[i])
    
    print("*" * 50)

    # print ("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    # eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    # exponential_experiment_result = exponential_experiment(extract(dataset), eps_values)
    # for i in range(len(eps_values)):
    #    print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])    


if __name__ == "__main__":

    main()
