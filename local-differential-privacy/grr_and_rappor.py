import math, random
import numpy as np

""" Globals """
DOMAIN = list(range(25)) # [0, 1, ..., 24]

""" Helpers """

def read_dataset(filename):
    """
        Reads the dataset with given filename.

        Args:
            filename (str): Path to the dataset file
        Returns:
            Dataset rows as a list.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result

# You can define your own helper functions here. #

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

def calculate_histogram(values):
    result = [0]*len(DOMAIN)
    for i in values:
        result[i] += 1
        
    return result


### HELPERS END ###

""" Functions to implement """

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    """
        Perturbs the given value using GRR protocol.

        Args:
            val (int): User's true value
            epsilon (float): Privacy parameter
        Returns:
            Perturbed value that the user reports to the server
    """
    d = len(DOMAIN)
    p = math.exp(epsilon) / (math.exp(epsilon) + d - 1)

    coin_toss = random.choices(["heads", "tails"], weights=[p, (1-p)], k =1)[0]
    
    if coin_toss == "heads":
        return val
    elif coin_toss == "tails":
        other_values = DOMAIN.copy()
        other_values.remove(val)
        return random.choice(other_values)
    

# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    """
        Estimates the histogram given GRR perturbed values of the users.

        Args:
            perturbed_values (list): Perturbed values of all users
            epsilon (float): Privacy parameter
        Returns:
            Estimated histogram as a list: [1.5, 6.7, ..., 1061.0] 
            for each hour in the domain [0, 1, ..., 24] respectively.
    """
    d = len(DOMAIN)
    p = math.exp(epsilon) / (math.exp(epsilon) + d - 1)
    q = (1-p) / (d-1)
    n = len(perturbed_values)

    reported_values = calculate_histogram(perturbed_values)
    
    estimated_values = [0]*d
    for i in range(d):
        estimated_values[i] = (reported_values[i] - (n*q)) / (p - q)

    return estimated_values


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    """
        Conducts the data collection experiment for GRR.

        Args:
            dataset (list): The daily_time dataset
            epsilon (float): Privacy parameter
        Returns:
            Error of the estimated histogram (float) -> Ex: 330.78
    """

    reported_data = []

    for true_value in dataset:  #simulating data collection
        reported_data.append(perturb_grr(true_value, epsilon))

    estimated_histogram = estimate_grr(reported_data, epsilon)
    true_histogram = calculate_histogram(dataset)

    return calculate_average_error(true_histogram, estimated_histogram)


# TODO: Implement this function!
def encode_rappor(val):
    """
        Encodes the given value into a bit vector.

        Args:
            val (int): The user's true value.
        Returns:
            The encoded bit vector as a list: [0, 1, ..., 0]
    """
    result = [0]*len(DOMAIN)
    result[val] = 1
    return result

# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    """
        Perturbs the given bit vector using RAPPOR protocol.

        Args:
            encoded_val (list) : User's encoded value
            epsilon (float): Privacy parameter
        Returns:
            Perturbed bit vector that the user reports to the server as a list: [1, 1, ..., 0]
    """
    result = [0]*len(DOMAIN)
    p = math.exp((epsilon/2)) / (math.exp((epsilon/2)) + 1)
    q = 1 / (math.exp((epsilon/2)) + 1)

    for i in range(len(encoded_val)):
        condition = random.choices(["preserve", "flip"], weights=[p, q], k=1)[0]
        
        if condition == "preserve":
            result[i] = encoded_val[i]
        elif condition == "flip":
            result[i] = int(not encoded_val[i])
        
    return result

# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    """
        Estimates the histogram given RAPPOR perturbed values of the users.

        Args:
            perturbed_values (list of lists): Perturbed bit vectors of all users
            epsilon (float): Privacy parameter
        Returns:
            Estimated histogram as a list: [1.5, 6.7, ..., 1061.0] 
            for each hour in the domain [0, 1, ..., 24] respectively.
    """
    p = math.exp((epsilon/2)) / (math.exp((epsilon/2)) + 1)
    q = 1 / (math.exp((epsilon/2)) + 1)
    n = len(perturbed_values)
    d = len(DOMAIN)

    perturbed_histogram = np.sum(perturbed_values, axis=0)
    estimated_values = [0]*d
    for i in range(d):
        estimated_values[i] = (perturbed_histogram[i] - (n*q)) / (p - q)

    return estimated_values
    
# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    """
        Conducts the data collection experiment for RAPPOR.

        Args:
            dataset (list): The daily_time dataset
            epsilon (float): Privacy parameter
        Returns:
            Error of the estimated histogram (float) -> Ex: 330.78
    """

    #encode users true data.
    true_encode = []
    for i in dataset:
        true_encode.append(encode_rappor(i))
    
    #apply perturb_rappor to true data.
    perturbed_encode = []
    for i in true_encode:
        perturbed_encode.append(perturb_rappor(i, epsilon))

    estimated_histogram = estimate_rappor(perturbed_encode, epsilon)
    true_histogram = calculate_histogram(dataset)

    return calculate_average_error(true_histogram, estimated_histogram)


def main():
    dataset = read_dataset("daily_time.txt")
    
    print("GRR EXPERIMENT")
    #for epsilon in [20.0]: 
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]: 
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)
    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))
    

if __name__ == "__main__":
    main()