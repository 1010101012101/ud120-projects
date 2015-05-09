#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    errors = [(predictions[i] - net_worths[i]) ** 2 for i in range(len(ages))]
    zipped = zip(errors, ages, net_worths)
    zipped.sort()
    zipped = zipped[:len(zipped) - len(zipped)/10]
    cleaned_data = [(age, net, error) for (error, age, net) in zipped]
    return cleaned_data

