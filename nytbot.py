
# coding: utf-8

# In[2]:

from __future__ import unicode_literals
from rtmbot.core import Plugin
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import HTMLParser as html
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[1]:

class myPlugin(Plugin):
    
    def __init__(self):
        top_df = []
        #msg = []
        
    
    def process_message(self, data):
        
        
        #text commands for cohlerbot
        if data['text'] == '!help':
            self.outputs.append(["C3ZMR09SN", 'Usage tips:\n1) "!headlines" - retrieves headlines\nTHEN\n2)"!news n" - retrieves summary of article at index n'])
            
        elif data['text'] == '!headlines':
            #retrieves story headlines from NYT website
            def get_headlines():
                response = requests.get("http://www.nytimes.com/")
                parser = BeautifulSoup(response.text, "html.parser")

                top_story = []
                secondary_story = []

                top_link = []
                secondary_link = []

                top_stories = parser.select('div.a-column')[0]
                top_headings = top_stories.select('h2.story-heading')

                for story in top_headings:
                    try:
                        top_story.append(story.get_text())
                        pattern = re.compile("http.+\.html")
                        head = re.search(pattern, str(story)).group(0)
                        head.replace('"','')
                        top_link.append(head)
                    except:
                        top_link.append(None)

                top_df = pd.DataFrame({'headline':top_story,'link':top_link})
                top_df.dropna(inplace=True)
                top_df.reset_index(inplace=True)
                top_df=top_df[['headline','link']]

                return top_df
            
            top_df = get_headlines()
            
            def make_headlines(top_df):
                ls = []
                for index,row in top_df.iterrows():
                    ls.append('>'+str(index)+') '+row['headline'])
                ls = '\n'.join(ls)
                headlines = u"*Headlines:*\n{}".format(ls)
                return headlines
            headlines = make_headlines(get_headlines())
            self.outputs.append(["C3ZMR09SN", headlines])
        
        elif data['text'][0:5] == '!news':
            
            story = int(data['text'][-((len(data['text'].strip())) - data['text'].strip().find(' ')):])

            def get_story(story):
                story_link = top_df.link[story]

                response = requests.get(story_link)
                parser = BeautifulSoup(response.text, "html.parser")

                #parse article to grab text, link, and title
                article_link = story_link

                title = top_df.headline[story]

                article = []

                for para in parser.select('p.story-body-text'):
                    article.append(para.get_text().replace(u'.\u201d', u'\u201d.')
                                                  .replace(u'!\u201d', u'\u201d!')
                                                  .replace(u'?\u201d', u'\u201d?')
                                                  .replace(u'\u25a0',''))

                #create string holding entire article text
                article = ' '.join(article).strip()
                return article, title, article_link



            #story input is an index 0 to n; where n is the number of headlines in top_df
            def summ(text):

                #tokes out any words with excess periods ('.') that may incorrectly break up sentences
                def no_abbv(article):
                    article_list = []
                    for word in article.split(' '):
                        if word.count('.') > 1 and word[-1] == '"':
                            article_list.append(word.replace('.',''))
                        if word.count('.') > 1:
                            article_list.append(word.replace('.',''))    
                        else:
                            article_list.append(word)

                    return ' '.join(article_list)

                #cleans text to alpha characters only
                def clean(text_list):
                    cleaned = []
                    for word in text_list:
                        try:
                            cleaned.append(re.sub('[^A-Za-z]+','', word))
                        except:
                            cleaned.append(word)

                    try:
                        return list(cleaned.remove(''))

                    except:
                        return cleaned

                #function to stem words
                def stemmer(text):
                    # initialize stemmer
                    stem1 = SnowballStemmer('english')
                    return clean([stem1.stem(word) for word in TextBlob(text).lower().words])

                # standardize feature scores
                def scaler(feature):
                    score = feature.values.reshape(-1,1)
                    scaler = StandardScaler()
                    scaler.fit(score)
                    return scaler.transform(score)

                article = no_abbv(text)

                #TFIDF for full text article
                stopwords = stemmer(' '.join(['mr','mrs','ms','said','a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'happen', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves']))

                def tf_idf(text, ngram_range):
                    vect = TfidfVectorizer(stop_words = stopwords, ngram_range=ngram_range)
                    tfidf = pd.DataFrame(vect.fit_transform([text]).toarray(), columns=vect.get_feature_names())

                    top_tokens = pd.DataFrame(pd.DataFrame(tfidf.sum().sort_values(ascending=False))).reset_index().rename(columns={'index':'token',0:'freq'})
                    return top_tokens

                top_tokens = tf_idf(article,ngram_range = (1,1))

                #processing article text into sentences
                def make_article_df(article):
                    tokens = TextBlob(article).sentences

                    sentences = []

                    for sentence in tokens:
                        sentences.append(str(sentence).decode('utf-8'))

                    #split article by sentences and put in df        
                    return pd.DataFrame({'sentence':sentences})

                article_df = make_article_df(article)

                #sentence importance scoring algorithm
                def sentence_score3(string_list,token_df):
                    score = []
                    #string_list = clean(string_list)
                    for word in stopwords:
                        if word in string_list[:]:
                            string_list.remove(word)
                    for index,row in token_df.head(5).iterrows():
                        for token1 in list(set(list(row.stem))):
                            for word in string_list:
                                if token1 in string_list:
                                    score.append(row.freq)
                                else:
                                    score.append(0)                                                 

                    #for word in stem_title:
                    #    if word in string_list:
                    #        score.append(1)
                    #    else:
                    #        score.append(0)

                    #return pd.Series(score).mean()#*len(sentence)
                    try:
                        return pd.Series(score).sum()/len(string_list)
                    except:
                        return 0

                #jaccard score for sentences for final sentence scoring
                def jaccard(summary,df):
                    jaccard_scores = []
                    summary = ' '.join(summary)

                    #summary = summary + ' ' + title

                    first = list(set(stemmer(summary)))
                    for word in stopwords:
                        if word in first[:]:
                            first.remove(word)
                    first = [x for x in first if x != '']
                    first = set(first)

                    for sentence in df.stem:
                        second = list(set(sentence))
                        for word in stopwords:
                            if word in second[:]:
                                second.remove(word)
                        second = [x for x in second if x != '']
                        second = set(second)
                        jaccard_scores.append((len(first & second) / float(len(first | second))))

                    return jaccard_scores

                def jaccard_keys(summary,df):
                    jaccard_terms = []
                    summary = ' '.join(summary)

                    #summary = summary + ' ' + title

                    first = list(set(stemmer(summary)))
                    for word in stopwords:
                        if word in first[:]:
                            first.remove(word)
                    first = [x for x in first if x != '']
                    first = set(first)

                    for sentence in df.stem:
                        second = list(set(sentence))
                        for word in stopwords:
                            if word in second[:]:
                                second.remove(word)
                        second = [x for x in second if x != '']
                        second = set(second)
                        jaccard_terms.append((first & second))

                    return jaccard_terms



                #stem_title = clean(stemmer(title)) 

                #stem column
                article_df['stem'] = article_df.sentence.apply(stemmer)

                top_tokens['stem'] = top_tokens.token.apply(stemmer)

                #score each sentence
                article_df['score'] = article_df.stem.apply(sentence_score3,token_df = top_tokens)

                #segment article by intro, body, and end
                position = []

                article_len = len(article_df) 
                round(article_len*.10)

                intro = [1]*int(round(article_len*.15))
                end = [3]*int(round(article_len*.33))
                body = [2]*int(round(article_len - (len(intro)+len(end))))

                position = intro + body + end

                #add position column in article df
                article_df['position'] = position

                #enumerate each sentence
                article_df.reset_index(inplace=True)
                article_df=article_df.rename(columns={'index':'num'})

                #construct list of sentences for summary using most important sentences from each position
                summary = [article_df.sort_values(by='score', ascending = False).sentence.iloc[0], 
                           article_df.sort_values(by='score', ascending = False).sentence.iloc[1]]

                #tfidf for first 2 lines picked
                summ_tokens = tf_idf(' '.join(summary),ngram_range = (2,2))
                summ_tokens['stem'] = summ_tokens.token.apply(stemmer)

                #creates summ_df and summary
                def make_summ():

                    first = article_df.sort_values(by='score', ascending = False).sentence.iloc[0]
                    second = article_df.sort_values(by='score', ascending = False).sentence.iloc[1]


                    summ_df1 = article_df[article_df.sentence != first]
                    summ_df2 = article_df[article_df.sentence != second]

                    summ_df = pd.merge(summ_df1[['sentence','num','score']],summ_df2[['sentence','num','score']])

                    summ_df['stem'] = summ_df.sentence.apply(stemmer)

                    return summ_df

                pos3_df = make_summ()
                pos3_df['score'] = pos3_df.stem.apply(sentence_score3, token_df = summ_tokens)

                pos3_df['jaccard_score'] = jaccard(summary,pos3_df)
                article_df['jaccard_score'] = jaccard(summary,article_df)
                pos3_df['jaccard_keys'] = jaccard_keys(summary,pos3_df)
                article_df['jaccard_keys'] = jaccard_keys(summary,article_df)

                avg = []
                for index,row in pos3_df.iterrows():
                    #avg.append(np.mean([row.score,row.jaccard_score]))
                    avg.append(row.score*row.jaccard_score)
                pos3_df['avg'] = avg

                article_df['score_scaled'] = scaler(article_df['score'])
                article_df['jaccard_scaled'] = scaler(article_df['jaccard_score'])
                pos3_df['score_scaled'] = scaler(pos3_df['score'])
                pos3_df['jaccard_scaled'] = scaler(pos3_df['jaccard_score'])
                pos3_df['avg_scaled'] = scaler(pos3_df['avg'])    

                #put summary sentences in order !!!!! needs work !!!!!    
                def order_summ():
                    summary = pd.DataFrame([article_df[article_df.num == pos3_df[pos3_df.avg_scaled < 5].sort_values(by='avg', ascending = False).num.iloc[0]][['sentence','num','score']].iloc[0],
                                            article_df.sort_values(by='score', ascending = False)[['sentence','num','score']].iloc[0], 
                                            article_df.sort_values(by='score', ascending = False)[['sentence','num','score']].iloc[1]])

                    if str(list(summary.sort_values(by='num').sentence)[0][0:4]).lower() == 'but ':
                        summ_list = [summary.sort_values(by='score',ascending=False).sentence.iloc[1],
                                     summary.sort_values(by='score',ascending=False).sentence.iloc[2],
                                     summary.sort_values(by='score',ascending=False).sentence.iloc[0]]

                    else:
                        summ_list = list(summary.sort_values(by='num').sentence)

                    sentence_list = list(summary.sort_values(by='num').num)

                    return (summ_list, sentence_list)


                summarized, sentence_list = order_summ()
                summarized = ' '.join(summarized)


                if summarized[-1] != '.':
                    summarized = summarized + '.'
                else:
                    pass

                summarized = summarized.replace(' ,',', ').replace(' .','.').replace(' \'s', '\'s').replace('( ','(').replace(' )',')').replace(' "','').replace(' :',':').replace(' ;',';')

                #compression
                summ_len = len(summarized)
                long_len = len(article)
                reduction = (1 - float(summ_len)/long_len)*100
                reduction_stat = 'Reduction Stats: [ Summary Length: ' + str(summ_len) + ' chars | Original Article Length: ' + str(long_len) + ' chars | ' + "%.2f" % reduction + '% redux ]'
                
                slack_summ = ['*' + title + '*','>'+summarized,reduction_stat]
                slack_summ = '\n'.join(slack_summ).encode('utf-8')
                
                return {'article_df':article_df, 
                        'top_tokens':top_tokens, 
                        'pos3_df':pos3_df, 
                        'summ_tokens':summ_tokens, 
                        'summary':summarized,
                        'reduction':reduction_stat,
                        'sentence_list':sentence_list,
                        'slack_summ':slack_summ}

            
            #top_df = get_headlines()
            if story in top_df.index:
                article, title, article_link = get_story(story)
                pkg = summ(article)
                self.outputs.append(["C3ZMR09SN", pkg['slack_summ']])
            
            else:
                msg = u'Headline {} is out of index, please select another headline.'.format(story)
                self.outputs.append(["C3ZMR09SN", msg])
            
        elif (data['text'][0] == u'!') and (len(data['text']) > 1):
            msg = u'Oops! Was that for me? I do not recognize the command: "{}", use "!help" for additional information.'.format(data['text'])
            self.outputs.append(["C3ZMR09SN", msg])
            
        else:
            pass


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



