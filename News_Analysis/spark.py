import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import matplotlib
from django.core.files.base import ContentFile
matplotlib.use("Agg")

# Word Cloud Functions


# def sent_TokenizeFunct(x):
#     return nltk.sent_tokenize(x)


def word_tokenize(x):
    splitted = [word for word in x.split()]
    return splitted


def removeStopWordsFunct(x):
    stop_words = ["a", " a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after",
                  "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along",
                  "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any",
                  "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear",
                  "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated",
                  "at", "available", "away", "awfully", "be", "became", "because", "become", "becomes", "becoming",
                  "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best",
                  "better", "between", "beyond", "both", "brief", "but", "by", "c'mon", "c's", "came", "can", "can't",
                  "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com",
                  "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing",
                  "contains", "corresponding", "could", "couldn't", "course", "currently", "definitely", "described",
                  "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing", "don't", "done", "down",
                  "downwards", "during", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough",
                  "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything",
                  "everywhere", "ex", "exactly", "example", "except", "far", "few", "fifth", "first", "five",
                  "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further",
                  "furthermore", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got",
                  "gotten", "greetings", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't",
                  "having", "he", "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby",
                  "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how",
                  "howbeit", "however", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in",
                  "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead",
                  "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "just", "keep",
                  "keeps", "kept", "know", "known", "knows", "last", "lately", "later", "latter", "latterly", "least",
                  "less", "lest", "let", "let's", "like", "liked", "likely", "little", "look", "looking", "looks",
                  "ltd", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more",
                  "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "nd", "near",
                  "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine",
                  "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now",
                  "nowhere", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones",
                  "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out",
                  "outside", "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed",
                  "please", "plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather",
                  "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively",
                  "respectively", "right", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see",
                  "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent",
                  "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six",
                  "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat",
                  "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup",
                  "sure", "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that",
                  "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there",
                  "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they",
                  "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly",
                  "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took",
                  "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "un", "under",
                  "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful",
                  "uses", "using", "usually", "value", "various", "very", "via", "viz", "vs", "want", "wants", "was",
                  "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were",
                  "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's",
                  "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
                  "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish",
                  "with", "within", "without", "won't", "wonder", "would", "wouldn't", "yes", "yet", "you", "you'd",
                  "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "zero", "b", "c", "d", "e",
                  "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
                  "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                  "T", "U", "V", "W", "X", "Y", "Z", "co", "op", "research-articl", "pagecount", "cit", "ibid", "les",
                  "le", "au", "que", "est", "pas", "vol", "el", "los", "pp", "u201d", "well-b", "http", "volumtype",
                  "par", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a1", "a2", "a3", "a4", "ab", "ac", "ad", "ae", "af",
                  "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw", "ax", "ay", "az", "b1", "b2", "b3", "ba", "bc",
                  "bd", "be", "bi", "bj", "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3", "cc",
                  "cd", "ce", "cf", "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp", "cq", "cr", "cs", "ct", "cu", "cv",
                  "cx", "cy", "cz", "d2", "da", "dc", "dd", "de", "df", "di", "dj", "dk", "dl", "do", "dp", "dr", "ds",
                  "dt", "du", "dx", "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej", "el", "em", "en", "eo",
                  "ep", "eq", "er", "es", "et", "eu", "ev", "ex", "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn",
                  "fo", "fr", "fs", "ft", "fu", "fy", "ga", "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy", "h2", "h3",
                  "hh", "hi", "hj", "ho", "hr", "hs", "hu", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib",
                  "ic", "ie", "ig", "ih", "ii", "ij", "il", "in", "io", "ip", "iq", "ir", "iv", "ix", "iy", "iz", "jj",
                  "jr", "js", "jt", "ju", "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc", "lf", "lj", "ln", "lo",
                  "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms", "mt", "mu", "n2", "nc", "nd", "ne", "ng", "ni", "nj",
                  "nl", "nn", "nr", "ns", "nt", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol", "om", "on",
                  "oo", "oq", "or", "os", "ot", "ou", "ow", "ox", "oz", "p1", "p2", "p3", "pc", "pd", "pe", "pf", "ph",
                  "pi", "pj", "pk", "pl", "pm", "pn", "po", "pq", "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra",
                  "rc", "rd", "rf", "rh", "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "rv", "ry",
                  "s2", "sa", "sc", "sd", "se", "sf", "si", "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy",
                  "sz", "t1", "t2", "t3", "tb", "tc", "td", "te", "tf", "th", "ti", "tj", "tl", "tm", "tn", "tp", "tq",
                  "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk", "um", "un", "uo", "ur", "ut", "va", "wa", "vd",
                  "wi", "vj", "vo", "wo", "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo",
                  "xs", "xt", "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"]
    filteredSentence = [w for w in x if not w in stop_words]
    return filteredSentence


# def lemmatizationFunct(x):
#     # nltk.download('wordnet')
#     lemmatizer = WordNetLemmatizer()
#     finalLem = [lemmatizer.lemmatize(s) for s in x]
#     return finalLem


# Spark Functions


def lemmatization_funct(y):
    lemmatizer = WordNetLemmatizer()
    finalLem = " ".join([lemmatizer.lemmatize(s) for s in y.split()])
    return finalLem


def less_then_four(l):
    text = " ".join([word for word in l.split() if len(word) > 3])
    return text


def remove_pun(text):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in text:
        if char not in punctuations:
            no_punct = no_punct + char

    return no_punct


def sparkfun(data, spark, model, Row):
    text = data[0]['text'].lower()
    no_punc = remove_pun(text)
    les = less_then_four(no_punc)
    lem = lemmatization_funct(les)
    dic = {"text_": lem}
    data[0].update(dic)
    row = spark.createDataFrame([Row(**i) for i in data])
    prediction = model[0].transform(row)
    text_rdd = prediction.select("text_").rdd.flatMap(lambda x: x)
    result = [row.asDict() for row in prediction.collect()]
    name = [d['name'] for d in result]
    figures = word_cloud(text_rdd, spark)
    text = [d['text'] for d in result]
    prediction = [d['prediction'] for d in result]
    news = ["Sport", "Entertainment", "Business", "Tech", "Politics"]
    # print(news[int(prediction[0])])
    n = news[int(prediction[0])]
    if int(prediction[0]) == 0:
        sport_prediction = model[1].transform(row)
        sport = [row.asDict() for row in sport_prediction.collect()]
        prediction1 = [d['prediction'] for d in sport]
        sports = ["Football", "Cricket", "Rugby", "Tennis", "Athletics"]
        s = sports[int(prediction1[0])]
        n = n + "/" + s

    return {"name": name, "text": text, "prediction": n, "word_cloud": figures[0], "word_frequency_graph": figures[1]}


def word_cloud(text_rdd, spark):
    # sentenceTokenizeRDD = text_rdd.map(sent_TokenizeFunct)
    # print("Text Rdd:")
    # print(text_rdd.collect())
    word_tokenize_rdd = text_rdd.map(word_tokenize)
    stopwordRDD = word_tokenize_rdd.map(removeStopWordsFunct)
    # lem_wordsRDD = stopwordRDD.map(lemmatizationFunct)
    # print("words")
    # print(stopwordRDD.collect())
    freqDistRDD = stopwordRDD.flatMap(lambda x: nltk.FreqDist(x).most_common()).map(lambda x: x).reduceByKey(
        lambda x, y: x + y).sortBy(lambda x: x[1], ascending=False)
    df_fDist = freqDistRDD.toDF()
    df_fDist.createOrReplaceTempView("myTable")
    df2 = spark.sql("SELECT _1 AS Keywords, _2 as Frequency from myTable limit 25")  # renaming columns
    pandD = df2.toPandas()  # converting spark dataframes to pandas dataframes
    fig = pandD.plot.barh(x='Keywords', y='Frequency', rot=1, figsize=(10, 8)).get_figure()
    fin = BytesIO()
    fig.savefig(fin, format='png', dpi=300)
    content_file_graph = ContentFile(fin.getvalue())
    # fig.savefig("bar2.jpg")
    # df2.show()
    wordcloudConvertDF = pandD.set_index('Keywords').T.to_dict('records')
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100,
                          colormap='Oranges_r').generate_from_frequencies(dict(*wordcloudConvertDF))
    # Dark2  Oranges_r
    plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    f = BytesIO()
    plt.savefig(f, format='png', dpi=300)
    content_file = ContentFile(f.getvalue())

    return [content_file, content_file_graph]

























# from pyspark import SparkContext
# from pyspark import SparkConf
# # from pyspark.sql import SparkSession
# from pyspark.ml import PipelineModel
# from pyspark.sql import SQLContext, Row
# # from functools import reduce
# import nltk
# from nltk.stem import WordNetLemmatizer
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# # import pandas as pd
# # import re
# # import string
#
# conf = SparkConf().setAppName("word_cloud").setMaster("local[4]")
# conf.set('spark.logConf', 'true')
# conf.set('spark.executor.memory', '2g')
# conf.set('spark.executor.cores', '2')
# conf.set('spark.cores.max', '2')
# conf.set('spark.driver.memory', '2g')
# sc = SparkContext(conf=conf)
# sc.setLogLevel("ERROR")
# spark = SQLContext(sc)
# # spark.sql("set spark.sql.shuffle.partitions=3")
# # sc = SparkSession \
# #                 .builder \
# #                 .master("spark://10.0.75.1:7077") \
# #                 .appName('word_cloud') \
# #                 .config("spark.executor.memory", '2g') \
# #                 .config('spark.executor.cores', '2') \
# #                 .config('spark.cores.max', '2') \
# #                 .config("spark.driver.memory", '2g') \
# #                 .getOrCreate()
# #
# # sc.sparkContext.setLogLevel("ERROR")
# spark.sql("set spark.sql.shuffle.partitions=3")
#
# model = PipelineModel.load("hdfs://localhost:19000/user/Waseem/Pipeline")
#
#
# def sent_TokenizeFunct(x):
#     return nltk.sent_tokenize(x)
#
#
# def word_tokenize(x):
#     splitted = [word for line in x for word in line.split()]
#     return splitted
#
#
# def removeStopWordsFunct(x):
#     # from nltk.corpus import stopwords
#     # stop_words = set(stopwords.words('english'))
#     stop_words = ["a", " a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after",
#                   "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along",
#                   "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any",
#                   "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear",
#                   "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated",
#                   "at", "available", "away", "awfully", "be", "became", "because", "become", "becomes", "becoming",
#                   "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best",
#                   "better", "between", "beyond", "both", "brief", "but", "by", "c'mon", "c's", "came", "can", "can't",
#                   "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com",
#                   "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing",
#                   "contains", "corresponding", "could", "couldn't", "course", "currently", "definitely", "described",
#                   "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing", "don't", "done", "down",
#                   "downwards", "during", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough",
#                   "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything",
#                   "everywhere", "ex", "exactly", "example", "except", "far", "few", "fifth", "first", "five",
#                   "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further",
#                   "furthermore", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got",
#                   "gotten", "greetings", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't",
#                   "having", "he", "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby",
#                   "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how",
#                   "howbeit", "however", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in",
#                   "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead",
#                   "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "just", "keep",
#                   "keeps", "kept", "know", "known", "knows", "last", "lately", "later", "latter", "latterly", "least",
#                   "less", "lest", "let", "let's", "like", "liked", "likely", "little", "look", "looking", "looks",
#                   "ltd", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more",
#                   "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "nd", "near",
#                   "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine",
#                   "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now",
#                   "nowhere", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones",
#                   "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out",
#                   "outside", "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed",
#                   "please", "plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather",
#                   "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively",
#                   "respectively", "right", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see",
#                   "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent",
#                   "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six",
#                   "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat",
#                   "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup",
#                   "sure", "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that",
#                   "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there",
#                   "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they",
#                   "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly",
#                   "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took",
#                   "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "un", "under",
#                   "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful",
#                   "uses", "using", "usually", "value", "various", "very", "via", "viz", "vs", "want", "wants", "was",
#                   "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were",
#                   "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's",
#                   "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
#                   "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish",
#                   "with", "within", "without", "won't", "wonder", "would", "wouldn't", "yes", "yet", "you", "you'd",
#                   "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "zero", "b", "c", "d", "e",
#                   "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
#                   "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
#                   "T", "U", "V", "W", "X", "Y", "Z", "co", "op", "research-articl", "pagecount", "cit", "ibid", "les",
#                   "le", "au", "que", "est", "pas", "vol", "el", "los", "pp", "u201d", "well-b", "http", "volumtype",
#                   "par", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a1", "a2", "a3", "a4", "ab", "ac", "ad", "ae", "af",
#                   "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw", "ax", "ay", "az", "b1", "b2", "b3", "ba", "bc",
#                   "bd", "be", "bi", "bj", "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3", "cc",
#                   "cd", "ce", "cf", "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp", "cq", "cr", "cs", "ct", "cu", "cv",
#                   "cx", "cy", "cz", "d2", "da", "dc", "dd", "de", "df", "di", "dj", "dk", "dl", "do", "dp", "dr", "ds",
#                   "dt", "du", "dx", "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej", "el", "em", "en", "eo",
#                   "ep", "eq", "er", "es", "et", "eu", "ev", "ex", "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn",
#                   "fo", "fr", "fs", "ft", "fu", "fy", "ga", "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy", "h2", "h3",
#                   "hh", "hi", "hj", "ho", "hr", "hs", "hu", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib",
#                   "ic", "ie", "ig", "ih", "ii", "ij", "il", "in", "io", "ip", "iq", "ir", "iv", "ix", "iy", "iz", "jj",
#                   "jr", "js", "jt", "ju", "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc", "lf", "lj", "ln", "lo",
#                   "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms", "mt", "mu", "n2", "nc", "nd", "ne", "ng", "ni", "nj",
#                   "nl", "nn", "nr", "ns", "nt", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol", "om", "on",
#                   "oo", "oq", "or", "os", "ot", "ou", "ow", "ox", "oz", "p1", "p2", "p3", "pc", "pd", "pe", "pf", "ph",
#                   "pi", "pj", "pk", "pl", "pm", "pn", "po", "pq", "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra",
#                   "rc", "rd", "rf", "rh", "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "rv", "ry",
#                   "s2", "sa", "sc", "sd", "se", "sf", "si", "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy",
#                   "sz", "t1", "t2", "t3", "tb", "tc", "td", "te", "tf", "th", "ti", "tj", "tl", "tm", "tn", "tp", "tq",
#                   "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk", "um", "un", "uo", "ur", "ut", "va", "wa", "vd",
#                   "wi", "vj", "vo", "wo", "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo",
#                   "xs", "xt", "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"]
#     filteredSentence = [w for w in x if not w in stop_words]
#     return filteredSentence
#
#
# def lemmatizationFunct(x):
#     nltk.download('wordnet')
#     lemmatizer = WordNetLemmatizer()
#     finalLem = [lemmatizer.lemmatize(s) for s in x]
#     return finalLem
#
#
# def sparkfun(data):
#     check = spark.createDataFrame([Row(**i) for i in data])
#     prediction = model.transform(check)
#     text_rdd = prediction.select("text").rdd.flatMap(lambda x: x)
#     word_cloud(text_rdd)
#     result = [row.asDict() for row in prediction.collect()]
#     name = [d['name'] for d in result]
#     text = [d['text'] for d in result]
#     prediction = [d['prediction'] for d in result]
#     return {"name": name, "text": text, "prediction": prediction}
#
#
# def word_cloud(text_rdd):
#     # print(text_rdd.collect())
#     sentenceTokenizeRDD = text_rdd.map(sent_TokenizeFunct)
#     # print("sentences")
#     # print(sentenceTokenizeRDD.collect())
#     word_tokenize_rdd = sentenceTokenizeRDD.map(word_tokenize)
#     # print("words")
#     # print(word_tokenize_rdd.collect())
#     stopwordRDD = word_tokenize_rdd.map(removeStopWordsFunct)
#     # print("stop words")
#     # print(stopwordRDD.collect())
#     lem_wordsRDD = stopwordRDD.map(lemmatizationFunct)
#     # print("Lemmatization")
#     # print(lem_wordsRDD.collect())
#     freqDistRDD = lem_wordsRDD.flatMap(lambda x: nltk.FreqDist(x).most_common()).map(lambda x: x).reduceByKey(
#         lambda x, y: x + y).sortBy(lambda x: x[1], ascending=False)
#     # print("Frequency")
#     print(freqDistRDD.collect())
#     df_fDist = freqDistRDD.toDF()  # converting RDD to spark dataframe
#     df_fDist.createOrReplaceTempView("myTable")
#     df2 = spark.sql("SELECT _1 AS Keywords, _2 as Frequency from myTable limit 25")  # renaming columns
#     pandD = df2.toPandas()  # converting spark dataframes to pandas dataframes
#     # fig = pandD.plot.barh(x='Keywords', y='Frequency', rot=1, figsize=(10, 8)).get_figure()
#     # fig.savefig("bar2.jpg")
#     # df2.show()
#     wordcloudConvertDF = pandD.set_index('Keywords').T.to_dict('records')
#     wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100,
#                           colormap='Dark2').generate_from_frequencies(dict(*wordcloudConvertDF))
#     # Dark2  Oranges_r
#     plt.figure(figsize=(14, 10))
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis('on')
#     plt.savefig("wordcloud")
#     # plt.show()
#
#
#
