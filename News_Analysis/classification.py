from pyspark.ml import PipelineModel

newsModel = PipelineModel.load("hdfs://localhost:19000/user/Waseem/bestPipeline")
sportsModel = PipelineModel.load("hdfs://localhost:19000/user/Waseem/sportsPipeline")
model = [newsModel, sportsModel]