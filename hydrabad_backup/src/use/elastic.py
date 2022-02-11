from elasticsearch import Elasticsearch
  
class Elastic:
    def __init__(self, user, password, endpoint):
        #user = "elastic"
        #password = "a73jhx59F0MC39OPtK9YrZOA"
        #endpoint = "https://e99459e530344a36b4236a899b32887a.westus2.azure.elastic-cloud.com:9243"
        self.es = Elasticsearch([endpoint], http_auth=(user, password))

    def send(index, doc):
        self.es.index(index=index, body=doc)