module Custom

using MLJ, MLJScikitLearnInterface, DataFrames

export MLJ, MLJScikitLearnInterface, transform

function transform(data, model)
    if "Species" in names(data)
        data = select!(data, Not(:Species))
    elseif "Grade 2014" in names(data)
        data = select!(data, Not(Symbol("Grade 2014")))
    elseif "class" in names(data)
        data = select!(data, Not(:class))
    end
    data = coalesce.(data, -99999)
    return data
end

end


# NOTE: In this model template DRUM is automatically loading "grade_regression.jlso"
#
# Some hooks are omitted here as they are implicitly generated by the DRUM library.
# This happens because DRUM knows how to work with many known types of models.
# They can be overriden in order to change the default behavior:
#
# function load_model(code_dir)
#     ...
# end
#
# function score(data, model; kwargs)
#     ...
# end
#
# See more about the built-in support of models:
# https://github.com/datarobot/datarobot-user-models/blob/master/DEFINE-INFERENCE-MODEL.md#built-in-model-support
