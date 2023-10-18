# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama

# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C

# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri fonksiyon aracılığıyla gösteriniz.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.width", None)
pd.set_option("display.max_columns", 700)
pd.set_option("display.max_rows", 700)
df = pd.read_csv("persona.csv")
def show_dataframe_info(dataframe):
    print("***** HEAD *****")
    print(df.head())
    print("***** TAIL *****")
    print(df.tail())
    print("***** SHAPE *****")
    print(df.shape)
    print("***** SIZE *****")
    print(df.size)
    print("***** INFO *****")
    print(df.info())
    print("***** DESCRIBE *****")
    print(df.describe().T)
    print(df.describe([0.05,0.25,0.5,0.75,0.90,0.95,0.99]).T)
    print("***** Any NAN? *****")
    print(df.isnull().sum().any())
    print("***** Total nan Values for Each Variable *****")
    print(df.isnull().sum())

show_dataframe_info(df)

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
print(df["SOURCE"].unique())  # sources
print(df["SOURCE"].nunique())  # number of unique source
print(df["SOURCE"].value_counts())  # frequencies of sources

# Soru 3: Kaç unique PRICE vardır?
print(df["PRICE"].nunique())  # number of unique price
print(df["PRICE"].unique())   # unique prices

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
print(df["PRICE"].value_counts())

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
print(df["COUNTRY"].value_counts())

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
print(df.groupby("COUNTRY").agg({"PRICE": "sum"}).sort_values("PRICE", ascending=False))  # 1.way
print(df.pivot_table("PRICE", "COUNTRY" ,aggfunc="sum").sort_values("PRICE", ascending=False))  # 2.way

# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?
print(df["SOURCE"].value_counts())  # 1.way
print(df.groupby("SOURCE").agg({"PRICE": "count"}))  # 2.way
print(df.pivot_table("PRICE", "SOURCE", aggfunc="count"))  # 3.way

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
print(df.groupby("COUNTRY").agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False))  # 1.way
print(df.pivot_table("PRICE", "COUNTRY", aggfunc="mean").sort_values("PRICE", ascending=False))  # 2.way

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
print(df.groupby("SOURCE").agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)) #1. yol
print(df.pivot_table("PRICE", "SOURCE", aggfunc="mean").sort_values("PRICE", ascending=False))  # 2.way

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
print(df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"}))  # 1.way
print(df.pivot_table("PRICE", "COUNTRY", "SOURCE" ,aggfunc="mean"))  # 2.way
# Note!: Çift kırılımlarda pivot_table kullanılması daha sağlıklı olabilir. Output daha düzgün şekilde

#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################
print(df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}))  # 1.way
print(df.pivot_table("PRICE", ["COUNTRY", "SOURCE", "SEX"], "AGE" ,aggfunc="mean"))  # 2.way
# Burada pivot table da kırılımları ayarlama (index ve columns) kişiye bağlıdır yukarıdaki gibi kırılım yapılırken aşağıdaki de bir seçenek olabilir
print(df.pivot_table(values="PRICE", index=["COUNTRY", "SOURCE"], columns=["AGE", "SEX"], aggfunc="mean"))

#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
print(agg_df)

#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
agg_df.reset_index(inplace=True) #inplace fonskyinou işlemin kalıcı olmasını sağlar
agg_df = agg_df.reset_index()  # 2.way
print(agg_df.head())


# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'
labels = [str(agg_df["AGE"].min()) + '_18', '19_30', '31_40', '41_' + str(agg_df["AGE"].max())]
agg_df["age_binned"] = pd.cut(agg_df["AGE"], [agg_df["AGE"].min(), 19, 30, 40, agg_df["AGE"].max()], labels=labels)

print(agg_df.info())

# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.

#agg_df["customers_level_based"] = agg_df["COUNTRY"] + "_" + agg_df["SOURCE"] + "_" + agg_df["SEX"] + "_" + agg_df["age_binned"]  # 2.way

agg_df["customers_level_based"] = agg_df[["COUNTRY", "SOURCE", "SEX", "age_binned"]].agg(lambda x: "_".join(map(str, x)).upper(), axis=1)

print(agg_df)

# Dikkat! list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18 Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df = agg_df.pivot_table("PRICE", "customers_level_based", aggfunc="mean")  # 2.way
agg_df.reset_index(inplace=True)
print(agg_df)
print(agg_df.shape)  # boyutunu burada kontrol ediyoruz. İstediğimiz sınıflandırma olmuşmu diye



# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız, ( 3 segment)
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,
label = ["C", "B", "A"]
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 3, labels=label)
#agg_df.reset_index(inplace=True)
print(agg_df)

# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
print(agg_df[agg_df["customers_level_based"] == "TUR_ANDROID_FEMALE_31_40"])
# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
print(agg_df[agg_df["customers_level_based"] == "FRA_IOS_FEMALE_31_40"])

#extra: customers_level_based değişkenine göre ortalama, max, min ve toplam price değerlerini ortalamaya göre büyükten küçüğe göre göster
print(agg_df.groupby("customers_level_based").agg({"PRICE": ["mean", "max", "min", "sum"]}).sort_values(("PRICE", "mean"), ascending=False))

# 2. way
print(agg_df.pivot_table("PRICE", "customers_level_based", aggfunc=["mean", "max", "min", "sum"]).sort_values(by=("mean", "PRICE"), ascending=False))


# Android ve ios kullanıcı sayılarını gösteriniz:
sns.countplot(data=df, x="SOURCE", palette="viridis")
plt.title(" Number of Anroid and IOS")
plt.grid()
print(plt.show())

# Ülke ve cinsiyet kırılımında Android ve ios kullancıı sayılarını gösteriniz

sns.countplot(data=df, x="SOURCE", hue="COUNTRY", palette="viridis")
plt.title(" Number of Android and IOS users By Country")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())

# Segmentlere göre Ortalama Fiyatları Gösteriniz:

sns.barplot(data=agg_df, x="SEGMENT", y="PRICE", estimator="mean", palette="viridis")
plt.title(" Average Price by Segments ")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())

# Segmentlere ve Cinsiyete göre Ortalama Fiyatları Gösteriniz:
sex = []
for i in agg_df["customers_level_based"]:
    if "FEMALE" in i:
        sex.append("FEMALE")
    else:
        sex.append("MALE")
agg_df["SEX"] = sex
print(agg_df)

sns.barplot(data=agg_df, x="SEGMENT", hue="SEX", y="PRICE", estimator="mean", palette="viridis")
plt.title("Avrage Price by Segments and Sex")
plt.grid()
plt.xticks(rotation=45)
print(plt.show())
