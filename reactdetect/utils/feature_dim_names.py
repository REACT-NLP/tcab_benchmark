from reactdetect.feature_extraction.feature_extractor import FeatureExtractor 
import pandas as pd

def get_feature_dim_names(feature_fcn_name):
    stub_kwargs = {
        'return_dict':True,
        'text_list':pd.Series(['hello world']),
        'lm_bert_model':None,
        'lm_bert_tokenizer':None,
        'lm_causal_model':None,
        'lm_causal_tokenizer':None,
        'lm_masked_model':None,
        'lm_masked_tokenizer':None,
        'labels':pd.Series([0]),
        'target_model':None,
        'device':None,
        'clean_extremes':True, 
        'output_text_list':None, 
        'save_extracted':False, 
        'unused_model_to_cpu':False
    }
    fe = FeatureExtractor(add_tags=['tm','lm','tp'])
    out = []
    corresponding_extractor_fcn= None 
    for f in fe.extractors:
         if f.__name__ == feature_fcn_name:
             #print(f.__name__,'==', feature_fcn_name)
             corresponding_extractor_fcn = f 
             #print(corresponding_extractor_fcn.__name__)
    import inspect
    fcn_params = [param.name for param in inspect.signature(corresponding_extractor_fcn).parameters.values()]
    # get subset of kwargs specifically for this function
    kwargs_ = {k: stub_kwargs[k] for k in fcn_params if k in stub_kwargs}
    try:
        corresponding_extractor_fcn(feature_list=out,*kwargs_)
    except Exception as e:
        pass 
    return out

if __name__ == '__main__':
    get_feature_dim_names('tp_bert')