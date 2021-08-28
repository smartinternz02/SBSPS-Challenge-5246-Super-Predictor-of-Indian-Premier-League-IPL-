
import joblib
import pandas as pd



def predictRuns(input_test):
    with open('GBR_regressor002.joblib','rb') as f:
        GBR_regressor002=joblib.load(f)
    with open('team_encoder002.joblib','rb') as f:
        team_encoder002=joblib.load(f)
    with open('venue_encoder002.joblib','rb') as f:
        venue_encoder002=joblib.load(f)


    
    
    
    

       
    test_case=pd.read_csv(input_test)
    
    test_case['venue']=venue_encoder002.transform(test_case['venue'].values.reshape(-1,1)).toarray()
    test_case['batting_team']=team_encoder002.transform(test_case['batting_team'])
    test_case['bowling_team']=team_encoder002.transform(test_case['bowling_team'])

    
    
    
    



    
      
    test_case=test_case[['venue','innings','batting_team','bowling_team']]
    
    score=GBR_regressor002.predict(test_case)
       
    return int(score)
