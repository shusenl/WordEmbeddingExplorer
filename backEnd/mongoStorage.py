from pymongo import MongoClient
#save to / read from database

# save table

class mongoStorage(object):
   def __init__(self):
     self.MONGODB_HOST = 'localhost'
     self.MONGODB_PORT = 27017
     self.DBS_NAME = 'wordEmbeddingPrecomputation'
     self.COLLECTION_NAME = 'randomDataProjPDF'
     #self.COLLECTION_NAME = 'randomProjPDF'

   def clear(self):
     client = MongoClient('mongodb://residue3.sci.utah.edu/')
     collection = client[self.DBS_NAME][self.COLLECTION_NAME]
     collection.remove({})

   def query(self, queryDict):
     #client = MongoClinet(self.MONGODB_HOST, self.MONGODB_PORT)
     client = MongoClient('mongodb://residue3.sci.utah.edu/')
     collection = client[self.DBS_NAME][self.COLLECTION_NAME]
     record = {}
     record = collection.find_one(queryDict)
     return record

   def save(self, saveDict):
     #client = MongoClient(self.MONGODB_HOST, self.MONGODB_PORT)
     client = MongoClient('mongodb://residue3.sci.utah.edu/')
     collection = client[self.DBS_NAME][self.COLLECTION_NAME]
     print saveDict
     collection.insert(saveDict)
     #insert_one only work with pymongo 2.9 and up
     #collection.insert_one(saveDict)
