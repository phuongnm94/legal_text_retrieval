 
# this command will not run if do not mount private test - and infer on public test
cp /data/private_test_question.json /app/data/zac2021-ltr-data/public_test_question.json

# this command will run infer the test file and get the output
cd /app && \
python3 src/infer.py && \
cp data/result_prediction.json /result/submission.csv
