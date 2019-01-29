
library(dplyr)
library(ggplot2)
library(tidytext)
library(janeaustenr)
library(stringr)
library(tm)
library(tokenizers)
library(data.table)
library(tm) #load text mining library
setwd() #sets R's working directory to near where my files are


NDT_RTIII_TEXT  <-Corpus
#specifies the exact folder where my text file(s) is for analysis with tm.
summary(NDT_RTIII_TEXT)

#Take two Corpuses, split them into sentences, combine Document title, combine lists#
tokensentences <- function(x) {
  data_frame(unlist(tokenize_sentences(x)))} #split x by sentences
sentences_split<-lapply(NDT_RTIII_TEXT,tokensentences) # apply tokensentences to Corpus
sentences <- do.call("rbind", sentences_split) #combine two lists into one
sentences$id <- rep(names(sentences_split), sapply(sentences_split, nrow)) #map document name to sentences
colnames(sentences)<-c("text","id")

allsentences <- sentences %>%
  group_by(id) %>% 
  mutate(linenumber = row_number()) %>% #assign line number
  mutate(chapter = cumsum(str_detect(text, regex("CHAPTER",ignore_case = FALSE)))) %>% #look for the words "Chapter" to assign chapter
  mutate(document = paste("Chapter",chapter,sep = "_")) %>% #assign chapter variable name
  filter(!chapter==23) %>%
  ungroup()

head(allsentences)
words <- allsentences %>%
  unnest_tokens(word,text) #sentences to words

data(stop_words)

#Word Frequency#

words <- words %>%
  filter(!word %in% stop_words$word,#removes stop words from the words file
         str_detect(word, "[a-z]"),
         !str_detect(word,"[1-9]")) #removes any word that has characters beyond a-z

words %>%
  count(word, sort = TRUE) #counts words

id_words <- allsentences %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word,#removes stop words from the words file
         str_detect(word, "[a-z]"),
         !str_detect(word,"[1-9]")) %>% #removes any word that has characters beyond a-z
  count(id,chapter, word, sort = TRUE) %>%
  ungroup()

total_words <- id_words %>% 
  group_by(id,chapter) %>% 
  summarize(total = sum(n))

id_words <- left_join(id_words, total_words)

id_words

freq_by_rank <- id_words %>% 
  group_by(id, chapter) %>% 
  mutate(rank = row_number(), 
         `term frequency` = n/total)

freq_by_rank

id_words <- id_words %>%
  bind_tf_idf(word, id, n)
id_words

id_words %>%
  select(-total) %>%
  arrange(desc(tf_idf))

#plot tf-idf by chapter#
id_words %>%
  filter(!id_words$id=="RT JTA Domains.txt") %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>% 
  group_by(chapter) %>% 
  top_n(3) %>% 
  ungroup %>%
  ggplot(aes(word, tf_idf, fill = id)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~chapter, ncol = 2, scales = "free") +
  coord_flip()


##Analyze Sentences and Bi/trigrams##
sentence_bigrams <- allsentences %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) #creates bigrams


library(tidyr)
bigrams_separated <- sentence_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ") #separates biagrams into two columns

bigrams_filtered <- bigrams_separated %>% #removes bigrams that contain a stop word
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>%
  filter(str_detect(word1, "[a-z]")) %>% #removes bigrams that contain a anything besides letters a-z
  filter(str_detect(word2, "[a-z]")) %>%
  filter(!str_detect(word1,"[1-9]")) %>%
  filter(!str_detect(word2,"[1-9]"))

# new bigram counts:
bigram_counts <- bigrams_filtered %>% 
  count(word1, word2, sort = TRUE)

bigrams_united <- bigrams_filtered %>%
  unite(bigram, word1, word2, sep = " ")

bigrams_united

bigram_tf_idf <- bigrams_united %>%
  count(document, bigram) %>%
  bind_tf_idf(bigram, document, n) %>%
  arrange(desc(tf_idf))

bigram_tf_idf

bigram_tf_idf %>%
  arrange(desc(tf_idf)) %>%
  mutate(bigram = factor(bigram, levels = rev(unique(bigram)))) %>% 
  group_by(document) %>% 
  top_n(10) %>% 
  ungroup %>%
  ggplot(aes(bigram, tf_idf, fill = document)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~document, ncol = 5, scales = "free") +
  coord_flip()

#graph bigrams#
library(igraph)
bigram_graph <- bigram_counts %>%
  filter(n > 20) %>%
  graph_from_data_frame()

bigram_graph

library(ggraph)
set.seed(2017)

ggraph(bigram_graph, layout = "fr") +
  geom_edge_link() +
  geom_node_point() +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1)

##analyzing bigrams##
bigrams_filtered %>%
  filter(word2 == "street") %>%
  count(id, word1, sort = TRUE)

trigram <- allsentences %>%
  unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word3 %in% stop_words$word) %>%
  count(word1, word2, word3, sort = TRUE)






######################
#####LDA Analysis#####
######################

# divide into documents, each representing one chapter
by_chapter <- allsentences

# split into words
by_chapter_word <- by_chapter %>%
  unnest_tokens(word, text)


word_counts <- words %>%
  anti_join(stop_words) %>%
  count(document, word, sort = TRUE) %>%
  ungroup()

chapters_dtm <- word_counts %>%
  cast_dtm(document, word, n)




library(topicmodels)
#Changed to 6 topics, because there are 6 domains in the JTA#
burnin <- 2000
iter <- 1000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE
chapters_lda_2 <- LDA(chapters_dtm, k = 2, control = list(seed = 1234))
chapters_lda_6 <- LDA(chapters_dtm, k = 6, method="Gibbs",control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))
chapters_lda_8 <- LDA(chapters_dtm, k = 8, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))
chapters_lda_10 <- LDA(chapters_dtm, k = 10, control = list(seed = 1234))
chapters_lda_22 <- LDA(chapters_dtm, k = 22, control = list(seed = 1234))

#perplexity is used to find the "best" model fit, the lowest perplexity is the better fitting model#
perplexity(chapters_lda_2)
perplexity(object=chapters_lda_6,newdata=chapters_dtm, control=list(iter=iter,thin=thin),use_theta=T,estimate_theta = TRUE)
perplexity(chapters_lda_8,newdata=chapters_dtm, control=list(iter=iter,thin=thin),use_theta=T,estimate_theta = TRUE)
perplexity(chapters_lda_10)
perplexity(chapters_lda_22)

#test several different levels of topics#
library(purrr)
n_topics <- c( 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20,30,40,50,60,70,80,90,100)
chapters_lda_compare <- n_topics %>%
  map(LDA, x = chapters_dtm, control = list(seed = 1234))


#plot the levels of perpexlity
library(tibble)
library(ggplot2)
data_frame(k = n_topics,
           perplex = map_dbl(chapters_lda_compare, perplexity)) %>%
  ggplot(aes(k, perplex)) +
  geom_point() +
  geom_line() +
  labs(title = "Evaluating LDA topic models",
       subtitle = "Optimal number of topics (smaller is better)",
       x = "Number of topics",
       y = "Perplexity")
#####

library(tidytext)
chapters_6_topics <- tidy(chapters_lda_6, matrix = "beta")
chapters_6_topics

chapters_8_topics <- tidy(chapters_lda_8, matrix = "beta")
chapters_8_topics

chapters_10_topics <- tidy(chapters_lda_10, matrix = "beta")
chapters_10_topics

chapters_22_topics <- tidy(chapters_lda_22, matrix = "beta")
chapters_22_topics

#select 10 highest betas by topic#

#6 topics#
top_terms_6 <- chapters_6_topics %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

#Graph them#
library(ggplot2)

top_terms_6 %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

#8 Topics 9-1 from the LSA, and recommended to revome the first dimension#
top_terms_8 <- chapters_8_topics %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

#Graph them#
library(ggplot2)

top_terms_8 %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

#10 Topics#
top_terms_10 <- chapters_10_topics %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

#Graph them#
library(ggplot2)

top_terms_10 %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

#23 Topics#
top_terms_22 <- chapters_22_topics %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

#Graph them#
library(ggplot2)

top_terms_23 %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()


#Now match chapters by gamma, This looks at what topics were associated with what chapters?#
#First we broke the book apart by randomly generated topics, now we reverse it and look at how the topics relate to the chapters#
chapters_gamma_6 <- tidy(chapters_lda_6, matrix = "gamma")
chapters_gamma_6

#probability of chapter per topic
chapters_gamma_6 <- chapters_gamma_6 %>%
  separate(document, c("id","chapter"), sep = "_", convert = TRUE)

chapters_gamma_6

chapters_gamma_6 %>%
  mutate(title = reorder(topic, gamma * topic)) %>%
  ggplot(aes(factor(topic), gamma)) +
  geom_boxplot() +
  facet_wrap(~ id)

chapter_classifications_6 <- chapters_gamma_6 %>%
  group_by( id, chapter) %>%
  top_n(1, gamma) %>%
  ungroup()

chapter_classifications_6

write.csv(chapter_classifications_6,"chapter_classifications_6.csv")

chapter_topics_6 <- chapter_classifications_6 %>%
  count(id, topic) %>%
  group_by(id) %>%
  top_n(1, n) %>%
  ungroup() %>%
  transmute(consensus =  id, topic)

chapter_classifications_6 %>%
  inner_join(chapter_topics_6, by = "topic") %>%
  filter(title != consensus)


#Split the data set into training and validation and compare perplexity#

burnin = 1000
iter = 1000
keep = 50

chapters_dtm
n <-nrow(chapters_dtm)
k <- 8 # number of topics

splitter <- sample(1:n, round(n * 0.50))
train_set <- chapters_dtm[splitter, ]
valid_set <- chapters_dtm[-splitter, ]

fitted <- LDA(train_set, k = k, method = "Gibbs",
              control = list(burnin = burnin, iter = iter, keep = keep) )
perplexity(fitted, newdata = train_set)
perplexity(fitted, newdata = valid_set)

#Next step is do the validation, with a single value of k, 
#but splitting the data into five so each row of the data gets a turn in the validation set. Here's how I did that:
library(doParallel)
folds <- 5
splitfolds <- sample(1:folds, n, replace = TRUE)

cluster <- makeCluster(detectCores(logical = TRUE) - 1) # leave one CPU spare...
registerDoParallel(cluster)

clusterEvalQ(cluster, {
  library(topicmodels)
})
clusterExport(cluster, c("chapters_dtm", "k", "burnin", "iter", "keep", "splitfolds"))

results <- foreach(i = 1:folds) %dopar% {
  train_set <- chapters_dtm[splitfolds != i , ]
  valid_set <- chapters_dtm[splitfolds == i, ]
  
  fitted <- LDA(train_set, k = k, method = "Gibbs",
                control = list(burnin = burnin, iter = iter, keep = keep) )
  return(perplexity(fitted, newdata = valid_set))
}
stopCluster(cluster)

#Finally, we now do this cross-validation many times, with different values of the number of latent topics to estimate. Here's how I did that:
  
  #----------------5-fold cross-validation, different numbers of topics----------------
cluster <- makeCluster(detectCores(logical = TRUE) - 1) # leave one CPU spare...
registerDoParallel(cluster)

clusterEvalQ(cluster, {
  library(topicmodels)
})

folds <- 5
splitfolds <- sample(1:folds, n, replace = TRUE)
candidate_k <- c(2, 3, 4, 6, 8, 9, 20, 30, 40, 50, 75, 100, 200, 300) # candidates for how many topics
clusterExport(cluster, c("chapters_dtm", "burnin", "iter", "keep", "splitfolds", "folds", "candidate_k"))

# we parallelize by the different number of topics.  A processor is allocated a value
# of k, and does the cross-validation serially.  This is because it is assumed there
# are more candidate values of k than there are cross-validation folds, hence it
# will be more efficient to parallelise
system.time({
  results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
    k <- candidate_k[j]
    results_1k <- matrix(0, nrow = folds, ncol = 2)
    colnames(results_1k) <- c("k", "perplexity")
    for(i in 1:folds){
      train_set <- chapters_dtm[splitfolds != i , ]
      valid_set <- chapters_dtm[splitfolds == i, ]
      
      fitted <- LDA(train_set, k = k, method = "Gibbs",
                    control = list(burnin = burnin, iter = iter, keep = keep) )
      results_1k[i,] <- c(k, perplexity(fitted, newdata = valid_set))
    }
    return(results_1k)
  }
})
stopCluster(cluster)

results_df <- as.data.frame(results)

ggplot(results_df, aes(x = k, y = perplexity)) +
  geom_point() +
  scale_x_continuous(breaks = c(2, 3, 4, 6, 8, 9, 10, 20, 30),limits=c(0,30)) +
  theme(axis.ticks.length=unit(1,"cm"))+
  geom_smooth(se = F) +
  ggtitle("5-fold cross-validation of topic modelling with the RT III Handbook",
          "(ie five different models fit for each candidate number of topics)") +
  labs(x = "Candidate number of topics", y = "Perplexity when fitting the trained model to the hold-out set")


#=====================Presentation of results===============
library(wordcloud)
library(tidyverse)
FinalModel <- LDA(chapters_dtm, k = 8, method = "Gibbs",
                  control = list(burnin = burnin, iter = iter, keep = keep) )

# approach to drawing word clouds of all topics from an object created with LDA,
n <- 100; palette = "Greens"; lda <- FinalModel

p <- posterior(lda)
w1 <- as.data.frame(t(p$terms)) 
w2 <- w1 %>%
  mutate(word = rownames(w1)) %>%
  gather(topic, weight, -word) 

pal <- rep(brewer.pal(9, palette), each = ceiling(n / 9))[n:1]

wd <- getwd()

# need to be careful for warnings that words didn't fit in, 
# in which case make the png device larger
# remove any PNG files sitting around in the temp folder
unlink("*.png")
# create a PNG for each frame of the animation
for(i in 1:ncol(w1)){
  png(paste0(i + 1000, ".png"), 8 * 100, 8 * 100, res = 100)
  par(bg = "grey95")
  w3 <- w2 %>%
    filter(topic == i) %>%
    arrange(desc(weight))
  with(w3[1:n, ], 
       wordcloud(word, freq = weight, random.order = FALSE, 
                 ordered.colors = TRUE, colors = pal))
  title(paste("RT III Topic", i))
  dev.off()
}


###LSA Trial###
textbookonly<-word_counts[!(word_counts$document=="RT JTA Domains.txt_1" |word_counts$document=="RT JTA Domains.txt_2" | word_counts$document=="RT JTA Domains.txt_3" |
                            word_counts$document=="RT JTA Domains.txt_4" | word_counts$document=="RT JTA Domains.txt_5" | word_counts$document=="RT JTA Domains.txt_6"),]
table(textbookonly$document)
textbookonly_dtm <- textbookonly %>%
  cast_dtm(document, word, n)

textbookonly_tdm <- as.matrix(t(textbookonly_dtm))

library(lsa)
txt_mat <- as.textmatrix(as.matrix(textbookonly_tdm))
txt_mat <- lw_logtf(txt_mat) * gw_idf(txt_mat)
lsa_model <- lsa(txt_mat,dims=dimcalc_share(share=0.5))

dim(lsa_model$tk) #Terms x New LSA Space
dim(lsa_model$dk) #Documents x New LSA Space
length(lsa_model$sk) #Singular Values
lsa_model_txtmat <- as.textmatrix(lsa_model)

print.textmatrix(lsa_model_txtmat,bag_lines=5,bag_cols=5)
#scree plot of singular values, skip the first dimension because it's correlated with document length
lsa_model_sk<-as.data.frame(lsa_model$sk)
lsa_model_sk$Dimensions<-c(1:9)
library(ggplot2)
ggplot(data=lsa_model_sk, aes(x=Dimensions,y=lsa_model$sk,group=1))+
         geom_line()+
         geom_point()+
         scale_x_continuous(breaks = c(1:9)) +
       labs(title = "Scree Plot of LSA SVD Values by Topic Numbers",
        x = "Number of topics",
        y = "Singular Values")



cosine(txt_mat) #compares similarly between documents, virtually identical to correlation

# MDS with raw term-document matrix compute distance matrix
# td.mat<- as.matrix(TermDocumentMatrix(textbookonly_tdm))
td.mat <- textbookonly_tdm
td.mat
dist.mat <- dist(as.matrix(textbookonly_dtm))
dist.mat  # check distance matrix

documents<-unique(word_counts$document)
documents<-c("RT Handbook.txt_13","RT Handbook.txt_7","RT Handbook.txt_1","RT Handbook.txt_12","RT Handbook.txt_18","RT Handbook.txt_9","RT Handbook.txt_6", 
             "RT Handbook.txt_23","RT Handbook.txt_21","RT Handbook.txt_16", "RT Handbook.txt_22", "RT Handbook.txt_15", "RT Handbook.txt_8",  "RT Handbook.txt_10",
             "RT Handbook.txt_2",  "RT Handbook.txt_19", "RT Handbook.txt_14", "RT Handbook.txt_11", "RT Handbook.txt_20", "RT Handbook.txt_5",  "RT Handbook.txt_17",
             "RT Handbook.txt_3",  "RT Handbook.txt_4")
documents<-as.data.frame(documents)


# MDS with LSA
td.mat.lsa <- lw_bintf(td.mat) * gw_idf(td.mat)  # weighting
lsaSpace <- lsa(td.mat.lsa)  # create LSA space
dist.mat.lsa <- dist(t(as.textmatrix(lsaSpace)))  # compute distance matrix
dist.mat.lsa  # check distance mantrix
txt.mat <- as.textmatrix(lsaSpace)
dkmat<-lsaSpace$dk

#Clster Dendrogram
plot(hclust(dist.mat.lsa, method="ward.D"))

plot(lsaSpace$sk,type = "o") #scree plot of singular values, skip the first dimension because it's correlated with document length

#Partitioning Aroud Medoids (More robust K-Means) silhouette plot#
pam_RT <- pam(dist.mat.lsa, 6,diss = T)
summary(pam_RT)
plot(pam_RT)

clusplot(pam(dist.mat.lsa, 8),lines = 0)

#K-means#
library(NbClust)
NbClust(data=dkmat,diss=dist.mat.lsa,distance=NULL,method="kmeans")



cosine(txt_mat) #compares similarly between documents, virtually identical to correlation


tkmatrix <- as.data.frame(lsaSpace$tk)


#plot tf-idf by chapter#
library(data.table)
tkmatrix <- setDT(tkmatrix, keep.rownames = TRUE)[]
setnames(tkmatrix, old = c('rn','V1','V2','V3','V4','V5','V6','V7','V8','V9'), new = c('term','D1','D2','D3','D4','D5','D6','D7','D8','D9'))
tkmatrix <- melt(tkmatrix, id=c("term"))


#sort 9 dimensions based on highest values
tkmatrix_mapped <- tkmatrix %>%
  arrange(desc(value)) %>%
  mutate(term = factor(term, levels = rev(unique(term)))) %>% 
  group_by(variable) %>% 
  top_n(10, n) %>%
  ungroup() 

write.csv(tkmatrix_mapped,"tkmatrix_mapped.csv")

#Plot 9 dimensions top positive terms 
tkmatrix %>%
  arrange(desc(value)) %>%
  mutate(term = factor(term, levels = rev(unique(term)))) %>% 
  group_by(variable) %>% 
  top_n(5) %>% 
  ungroup %>%
  ggplot(aes(term, value, fill = variable)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "value") +
  facet_wrap(~variable, ncol = 2, scales = "free") +
  coord_flip()


###Heatmap of dimensions###
dkmatrix <- round(as.data.frame(lsaSpace$dk),6)
dkmatrix 
rownames(dkmatrix) <- c("13","7","1","12","18","9","6",
                        "21","16","22","15","8","10",
                        "2","19","14","11","20","5","17","3","4")
setnames(dkmatrix, old = c('V1','V2','V3','V4','V5','V6','V7','V8','V9'), new = c('D1','D2','D3','D4','D5','D6','D7','D8','D9'))

dkmatrixscaled <- as.matrix(scale(dkmatrix))
heatmap(t(dkmatrixscaled), Colv=NA,Rowv=NA, scale='none')

library(lattice)
library("RColorBrewer")
brewer.div <- colorRampPalette(brewer.pal(11, "Spectral"), interpolate = "spline")
levelplot (dkmatrixscaled, col.regions = brewer.div(200), aspect = "iso",scale=list(x=list(rot=45)))

library(ggplot2)
library(reshape2)

rnames<- rownames(dkmatrix)
dkmatrix<-cbind(rnames,data.frame(dkmatrix))
dkmatrix.m<-melt(dkmatrix)

(p <- ggplot(dkmatrix.m, aes(rnames, variable)) +
    geom_tile(aes(fill = value))+
    scale_fill_distiller(palette = "Spectral")+
    # scale_fill_gradient(low = "white",high = "steelblue") +
    scale_x_discrete("Chapter",breaks=c(1:23), labels=c(1:23),limits=c(1:23)) +
    labs(x = "Dimesions", y = "Chapters"))
  

##Distance heatmap##
cosinemat <- cosine(txt_mat)
cosinemat<-round(cosinemat,2)

library(reshape2)
melted_cosinemat <- melt(cosinemat)
head(melted_cosinemat)


library(ggplot2)
ggplot(data = melted_cosinemat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()

# Get upper triangle of the correlation matrix
get_upper_tri <- function(cosinemat){
  cosinemat[lower.tri(cosinemat)]<- NA
  return(cosinemat)
}

upper_tri <- get_upper_tri(cosinemat)
upper_tri

library(reshape2)
melted_cosinemat <- melt(upper_tri, na.rm = TRUE)
# Heatmap
library(ggplot2)
ggplot(data = melted_cosinemat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

reorder_cormat <- function(cosinemat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cosinemat)/2)
  hc <- hclust(dd)
  cosinemat <-cosinemat[hc$order, hc$order]
}

# Reorder the correlation matrix
cosinemat <- reorder_cormat(cosinemat)
upper_tri <- get_upper_tri(cosinemat)
# Melt the correlation matrix
melted_cosinemat <- melt(upper_tri, na.rm = TRUE)
# Create a ggheatmap
ggheatmap <- ggplot(melted_cosinemat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0.18, limit = c(0,.36), space = "Lab", 
                       name="Cosine\nSimilarity") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()
# Print the heatmap
print(ggheatmap)

ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

# MDS
fit <- cmdscale(dist.mat.lsa, eig = TRUE)
points <- data.frame(x = fit$points[, 1], y = fit$points[, 2])
ggplot(points, aes(x = x, y = y)) + geom_point(data = points, aes(x = x, y = y, 
                                                                      color = df$view)) + geom_text(data = points, aes(x = x, y = y - 0.2, label = 
                                                                                                                           row.names(df)))

fit <- cmdscale(dist.mat.lsa, eig = TRUE, k = 3)
colors <- rep(c("blue", "green", "red"), each = 3)
scatterplot3d(fit$points[, 1], fit$points[, 2], fit$points[, 3], color = colors, 
              pch = 16, main = "Semantic Space Scaled to 3D", xlab = "x", ylab = "y", 
              zlab = "z", type = "h")

