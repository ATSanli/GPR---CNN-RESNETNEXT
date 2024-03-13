import pickle


# Veri yükleme
with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Anahtarları yazdırma
print(train_data.columns)

# X ve y'yi belirleme
X = train_data['data']
y = train_data['labels']
