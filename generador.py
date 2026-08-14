import faiss as fb
import pymupdf4llm
import re
from sentence_transformers import SentenceTransformer
import json

#Abre las preguntas 
prego=pymupdf4llm.to_text("Extracto_Preguntas_50_v2.pdf")
#Busca las preguntas tomando como referencia ¿?
interroga=r'¿(.*?)\?'
preguntas1=re.findall(interroga,prego,re.DOTALL)
preguntas=[]
for item in preguntas1:
    preguntas.append("query: ¿" + item.strip() +"?")

#Abre y lee el indice FAISS
doc='indice.faiss'
indicefaiss=fb.read_index(doc)

#Abre y carga la metadata 
Metadatatotal=[]
with open("Metada.jsonl",'r',encoding='utf-8') as file:
    for linea in file:
        Metadatatotal.append(json.loads(linea))
        
#Carga el transformer para desencriptar
transformer='intfloat/multilingual-e5-small'
modelot=SentenceTransformer(transformer)

#Convertir la pregunta en vector
vpregunta=[]
for i in range(0,len(preguntas)):
    vpregunta.append(modelot.encode([preguntas[i]],convert_to_numpy=True,normalize_embeddings=True))
#Determinación de los fragmentos más relevantes
top10=10
distancias=[]
indices=[]
for j in range(0,len(preguntas)):
    dist,ind=indicefaiss.search(vpregunta[j],top10)
    distancias.append(dist)
    indices.append(ind)

#Búsqueda de documentos relevantes
top100=100  
distanciasdocs=[]
indicesdocs=[]
for j in range(0,len(preguntas)):
    dist,ind=indicefaiss.search(vpregunta[j],top100)
    distanciasdocs.append(dist)
    indicesdocs.append(ind)

#Guarda los resultados
docmejores=3
arch="resultados.jsonl"
with open(arch,"w",encoding="utf-8") as f:
    for lo in range(len(preguntas)):
        query_id=f"q{lo + 1:03d}" 
        #Determinar los 3 mejores documentos
        docrevisados=set()
        documentosjson=[]
        rangodoc=1
        
        for i,idx in enumerate(indicesdocs[lo][0]):
            if idx!=-1 and idx<len(Metadatatotal):
                resultado=Metadatatotal[idx]
                docid=resultado['doc_id'] 
                
                if docid not in docrevisados:
                    docrevisados.add(docid)
                    documentosjson.append({
                        "rank": rangodoc,
                        "doc_id": docid
                    })
                    rangodoc+=1                    
                if len(documentosjson)>=docmejores:
                    break

        #Para fragmentos
        fragmentosjson=[]
        rangofrag=1
        
        for kl in range(top10):
            idxfrag=indices[lo][0][kl] 
            
            if idxfrag!=-1 and idxfrag<len(Metadatatotal):
                metadatafrag=Metadatatotal[idxfrag]
                fragmentosjson.append({
                    "rank": rangofrag,
                    "chunk_id": metadatafrag['chunk_id'],
                    "doc_id": metadatafrag['doc_id'],
                    "text": metadatafrag['texto']
                })
                rangofrag+=1
                
        #Se une todo
        totales={
            "query_id": query_id,
            "documents": documentosjson,
            "fragments": fragmentosjson}
        
        lineaj= json.dumps(totales,ensure_ascii=False)
        f.write(lineaj + '\n')
