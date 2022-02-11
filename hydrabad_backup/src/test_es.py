
from datetime import datetime
from elasticsearch import Elasticsearch

user = "elastic"
password = "a73jhx59F0MC39OPtK9YrZOA"
endpoint = "https://e99459e530344a36b4236a899b32887a.westus2.azure.elastic-cloud.com:9243"
es = Elasticsearch([endpoint], http_auth=(user, password))
index = 'gmtc_searcher'

#doc = {
#    'author': 'kimchy',
#    'text': 'Elasticsearch: cool. bonsai cool.',
#    'timestamp': datetime.now(),
#}
#res = es.index(index=index, body=doc)

es.indices.refresh(index=index)
res = es.search(index=index, body={
    "size": 100,
    #"sort": {"time": "desc"},
    #"query": {"match_all": {}}
    "query": {"match": {'algo':'anpr'}}
    })

print("Got %d Hits:" % res['hits']['total']['value'])
for hit in res['hits']['hits']:
    print(hit['_source'])

#es.delete_by_query(index='gmtc_searcher', body={"query": {"match_all": {}}})