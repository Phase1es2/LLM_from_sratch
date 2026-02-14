import math

def xavier_init(in_features: int, out_features: int, ):
    return math.sqrt(2.0 / (in_features + out_features))