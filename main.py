from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

from xgboost.spark import SparkXGBClassifier

spark_xgb_classifier = SparkXGBClassifier(
  features_col="features",
  label_col="label",
  #num_workers=2,
  device='cpu',
)

spark = SparkSession.builder\
                    .master('local[*]')\
                    .appName('bank_fraud_detection')\
                    .getOrCreate()

bank_data = spark.read.csv('data/Base.csv', header=True, inferSchema=True, nullValue='NA')

bank_data = bank_data.withColumn("label", bank_data.fraud_bool.cast('float')).drop('fraud_bool')

bank_data = bank_data.dropna()

bank_data_train, bank_data_test = bank_data.randomSplit([0.8, 0.2], seed=42)

indexers = StringIndexer(inputCols=['payment_type', 'employment_status', 'housing_status', 'source', 'device_os'], outputCols=['payment_type_index', 'employment_status_index', 'housing_status_index', 'source_index', 'device_os_index'])

assembler = VectorAssembler(inputCols=['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count', 
                                       'customer_age', 'days_since_request', 'intended_balcon_amount', 'zip_count_4w', 'velocity_6h',
                                       'velocity_24h', 'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 
                                       'credit_risk_score', 'email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'bank_months_count', 
                                       'has_other_cards', 'proposed_credit_limit', 'foreign_request', 'session_length_in_minutes', 
                                       'keep_alive_session', 'device_distinct_emails_8w', 'device_fraud_count', 'month', 'payment_type_index',
                                       'employment_status_index', 'housing_status_index', 'source_index', 'device_os_index'], outputCol='features')

#model = DecisionTreeClassifier()
model = spark_xgb_classifier

pipeline = Pipeline(stages=[indexers, assembler, model])

pipeline = pipeline.fit(bank_data_train)

prediction = pipeline.transform(bank_data_test)

evaluator = BinaryClassificationEvaluator()

print("areaUnderROC", evaluator.evaluate(prediction, {evaluator.metricName: "areaUnderROC"}))

spark.stop()