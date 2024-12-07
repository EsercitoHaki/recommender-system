import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request

app = Flask(__name__)

# Kết nối tới database
conn = mysql.connector.connect(
    host="localhost",
    user="root",  
    password="",
    database="webbanhang"
)

try:
    # Truy vấn dữ liệu từ bảng products
    query = 'SELECT * FROM products'

    df_sanpham = pd.read_sql(query, conn)
    print(df_sanpham.head())

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    # Đóng kết nối
    conn.close()

features = ['name', 'price']

def combineFeatures(row):
    return str(row['price']) + " " +str(row['name'])

df_sanpham['combineFeatures'] = df_sanpham.apply(combineFeatures, axis=1)

print(df_sanpham['combineFeatures'])

tf = TfidfVectorizer()
tfMatrix = tf.fit_transform(df_sanpham['combineFeatures'])

similar = cosine_similarity(tfMatrix)

#print(similar)

number = 5
@app.route('/api', methods=['GET'])
def recommend_san_pham():
    ket_qua = []
    productId = request.args.get('id')
    productId = int(productId)

    if productId not in df_sanpham['id'].values:
        return jsonify({'loi':'Id khong hop le'})
    
    indexProduct = df_sanpham[df_sanpham['id'] == productId].index[0]

    similarProduct = list(enumerate(similar[indexProduct]))

    # for indexProduct, score in similarProduct:
    #     product_name = df_sanpham.loc[indexProduct, 'name']
    #     print(f"Tên sản phẩm: {product_name}, Độ tương đồng: {score}")

    #print(similarProduct)

    sortedSimilarProduct = sorted(similarProduct, key=lambda x:x[1], reverse=True)

    def lay_id(index):
        return (df_sanpham[df_sanpham.index == index]['id'].values[0])

    for i in range(1, number + 1):
        ket_qua.append(int(lay_id(sortedSimilarProduct[i][0])))

    return ket_qua

if __name__ == '__main__':
    app.run(port=5555)