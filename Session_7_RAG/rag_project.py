# ******************  IMPORT LIBRARIES  ******************
import pandas as pd 
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# ******************  IMPORT DATA  ******************
with open('clean_text.txt', "r", encoding="utf-8") as f:
    text = f.read()

# ******************  CLAEAN TEXT  ******************
clean_text = re.sub(r'\s+', ' ', text.replace('\u200c', ' ')).strip()

# ******************  Hyperparameters  ******************

chunk_len = 500    #512 =>1536
overlap   = 0.2
# ******************  CHUNKING  ******************
def chunk_text(text, max_len, overlap):
    chunks = []
    start = 0
    text_len = len(text)
    overlap_len = int(max_len * overlap)

    while start < text_len:
        end = start + max_len
        #if last parageraph longer than text len 
        if end >= text_len:
            chunk = text[start:]
            chunks.append(chunk.strip())
            break
        
        #find a last space for start and end a chunk
        last_space = text.rfind(' ', start, end)
        if last_space == -1 or last_space <= start:
            last_space = end

        chunk = text[start:last_space].strip()
        chunks.append(chunk)

        # find a last space for overlab and starting a new chunk
        start = last_space - overlap_len
        if start < 0:
            start = 0
        else:
            
            prev_space = text.rfind(' ', 0, start)
            if prev_space != -1:
                start = prev_space + 1  

    return chunks

chunks = chunk_text(clean_text, max_len=chunk_len, overlap=overlap)
df = pd.DataFrame({'chunk': chunks})
df['length'] = df['chunk'].apply(len)            # make a dataframe with chunk column and long of each chunk

# ******************  IMPORT EMBEDING MODEL  ******************
model =SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2', cache_folder='./DistiluseBaseMultilingualCasedV2')
embeddings = model.encode(chunks, show_progress_bar=True)

# ******************  EMBEDIG EACH ROW  ******************
df['embeddings'] = list(embeddings)
#print(df)

# ******************  FAISS INIT  ******************
'''dim = embeddings[0].shape[0]
print ('dimon of each embedding is :',dim)
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))
print('for_search :',index)'''
# ******************  TAKE A USER QUERY  ******************
user_query = "معرفی سرویس ابری رایابات: دستیار هوشمند فروش گفتگومحور مبتنی بر مدل های زبانی بزرگ شرکت شما به دنبال تحول در فرآیندهای فروش و تعامل با مشتریان است؟ رایابات به عنوان یک سرویس ابری پیشرفته، امکان طراحی و پیاده سازی عامل های هوشمند فروش گفتگومحور را با استفاده از آخرین فناوری های پردازش زبان طبیعی (NLP) و مدل های زبانی بزرگ (LLM) فراهم می کند. این راهکار، نه تنها تعامل با مشتریان را بهینه می سازد، بلکه با ابزارهای تحلیلی قدرتمند و داشبوردهای کاربردی، مدیریت فرآیند فروش را متحول می کند. مسئله بازار"
query_embedding = model.encode([user_query])[0]

# ******************  PADDING FOR 512 => 1536  ******************
def pad_to_1536(embedding):

    embed_array = np.array(embedding)
    pad_length = 1536 - len(embed_array)
    
    if pad_length > 0:
        padded = np.pad(embed_array, (0, pad_length), mode='constant')
    else:
        padded = embed_array[:1536]  # truncate if too long

    return padded


# ******************  FAISS RESULT  ******************
'''D, I = index.search(np.array([query_embedding]), k=3)
result = [chunks[i] for i in I[0]]'''
# ******************  SCORES  ******************
def cosine_similarity(vec1, vec2):
    'cosine_similarity'
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def l2_distance(vec1, vec2):
    "l2_distance"
    return np.linalg.norm(vec1 - vec2)

def l1_distance(vec1, vec2):
    'l1_distance'
    return np.sum(np.abs(vec1 - vec2))
    
def retrieve_best_chunk(df, user_embedding ):

    cosine_scores ,l2_scores ,l1_scores = [] ,[] ,[]
    for _, row in df.iterrows():
        chunk_embedding = np.array(row['padded_embedding'])
         
        cosine = cosine_similarity(user_embedding, chunk_embedding) #maximum score
        l2 = l2_distance(user_embedding, chunk_embedding) #minimum distance
        l1 = l1_distance(user_embedding, chunk_embedding) #minimum distance
        
        cosine_scores.append(cosine)
        l2_scores.append(l2)
        l1_scores.append(l1)

    df['cosine_score'] = cosine_scores
    df['l2_score'] = l2_scores
    df['l1_score'] = l1_scores        
    return df

# ******************  TAKE A RESULT AS A EXCEL FILE  ******************
pad_df = df.copy()
pad_df['padded_embedding'] = df['embeddings'].apply(pad_to_1536)
pad_query_embedding = pad_to_1536(query_embedding)

final_df = retrieve_best_chunk(pad_df, pad_query_embedding  ) #padded_embedding=1536  & embeddings=512

final_df['embeddings'] = final_df['embeddings'].apply(lambda x: ','.join(map(str, x)))
final_df['padded_embedding'] = final_df['padded_embedding'].apply(lambda x: ','.join(map(str, x)))

final_df.to_csv('result4_like_result1.csv' , index=False , encoding='utf-8-sig')
