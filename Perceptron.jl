module Perceptron
export SingleLayerPerceptron

using Base.Test
using Logging

function sigmoid(x::Number, offset::Float64=0.5)
    return(1/(1 + exp(-(x-offset))))
end
@test sigmoid(0.5, 0.5) == 0.5 # poor testing, I know ¬.¬

function threshold(x, t=0.5)
    return(x < t ? 0 : 1)
end

type SingleLayerPerceptron
    weights::Array{Float64,1}
    train!::Function
    predict::Function
    decision::Function
    function SingleLayerPerceptron(num_features, decision_function=threshold)
        self = new()
        self.weights = zeros(num_features+1)
        self.train! = (X, y) -> train!(self, X, y)
        self.predict = (X) -> predict(self, X)
        self.decision = decision_function
        return(self)
    end
end

function train!(perceptron::SingleLayerPerceptron, train_features::Array, class_feature::Array, learning_rate::Float64=0.1)
    for i=1:size(train_features,1)
        x = hcat(1, train_features[i,:])
        result = perceptron.predict(train_features[i,:])
        debug(result)
        perceptron.weights = perceptron.weights+ learning_rate*x'*(class_feature[i] - [result])
        info(perceptron.weights)
    end
end

function predict(perceptron::SingleLayerPerceptron, test_features::Array)
    x = hcat(1, test_features) # add offset term
    float_prediction = x*perceptron.weights
    return(perceptron.decision(float_prediction[1]))
end

end # end of module
#====================#
