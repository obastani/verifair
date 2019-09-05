import pandas as pd
import feature_engineering_func as fe_func
import cnn_func

categories = [
    'cat',
    'tiger',
    'lion',
    'dog'
]

file_paths = [
    # "/Users/xin/repos/fairness-verification/data/quickdraw/full-raw-" + x + ".ndjson" for x in categories
    "../../../data/quickdraw/" + x + ".ndjson" for x in categories
]

def main():
    dfs = []
    for i in range(len(categories)):
        cat = categories[i]
        df = pd.read_json(file_paths[i], lines=True)
        fe_func.feature_engineering_CNN(df, cat, 60000, 'word')
        df1 = pd.read_pickle('./data/'+cat+'.pkl')
        dfs.append(df1)
    	
    # the original author hardcoded the number of categories to be four, but it shouldn't be hard to change
    assert(len(dfs) == 4)

    ip_model, X_tr, y_tr, X_te, y_te = cnn_func.CNN_image_recognition(dfs[0], dfs[1], dfs[2], dfs[3], sample=60000,
                                                                      binary=False, \
                                                                      convlayer=64, neuron=100, batchsize=128, epoch=20)

    score = ip_model.evaluate(X_te, y_te)

    print('Testing score: '+str(score))

    model_path = "../../dis_model"
    print('Saving the model to '+model_path)
    ip_model.save(model_path)

if __name__ == '__main__':
    main()
