library(tidyverse) 
library(ggplot2)
library(stringr)
library(caret)
library(quanteda)
library(doSNOW)
library(e1071)
library(irlba)
library(kernlab)
library(tidytext)
library(textdata)
library(keras)
library(wordcloud)
library(wordcloud2)
library(reshape2)
library(tm)
library(igraph)
library(ggraph)

input_file <- "data/Gungor_2018_VictorianAuthorAttribution_data-train.csv"

# loading in the dataset
read_data <- function(num) {
  dataset <- read.csv(input_file, header = T, stringsAsFactors = FALSE, nrows = num)
}

# sampling 2 authors
df <- read_data(1294)

# sampling 3 authors
df <- read_data(1507)

# Check for missing data: we have no missing data
colSums(is.na(df))  

# rename author column to label and convert to factor
df <- df %>% rename(label = author) 
df$label <- as.factor(df$label)

# Feature engineering - calculate character length
char_length <- function(df, text) {
  df %>% mutate(char_length = str_count(text, pattern = boundary(type = "character")))
}
df <- char_length(df, text)

# Probability class distribution
prob_dist <- function(x) {
  prop.table(summary(x))
}
prob_dist(df$label)

# Visualization: distribution between Authors and Character length 
ggplot(df, aes(x = char_length, fill = label)) +
  scale_fill_discrete(name = "Authors", labels = c("Arthur Conan Doyle", "Charles Darwin")) + geom_histogram(binwidth = 10) + theme_bw() +
  labs(y = "Author ID", x = "Char Length",  title="Frequency of Character Length with Author's Label", caption = "Data from UCI Machine learning Repository")
ggsave("plots/plot1.png", plot = last_plot())

# changing author name
authors_name <- function(df, label) {
  df <- df %>% mutate(author_name = case_when(label == 1 ~ "Arthur Conan Doyle", label == 2 ~ "Charles Darwin", TRUE ~ "Missing" ))
}
df <- authors_name(df, label)

# Authors with the highest text length  
df %>% group_by(author_name) %>% tally(sort = TRUE)
# class inbalance
df %>% group_by(label) %>% tally(sort=TRUE)

# stratified sampling
split <- function(data) {
  splits <- sample(1:3, size = nrow(df), prob = c(.5, .2, .3), replace = T)
  training <<- df[splits == 1,]
  validation <<- df[splits == 2,]
  testing <<- df[splits == 3,]
}
split(df)

# class distribution
prob_dist(training$label)
prob_dist(validation$label)
prob_dist(testing$label)

# shuffle data: in order to reduce variance
shuffleRows <- function(df){
  return(df[sample(nrow(df)),])
}

training <- shuffleRows(training)
validation <- shuffleRows(validation)
testing <- shuffleRows(testing)

#------------------------------------------------------------------------------------------------------------------------------------------------------
# Data preparation for sentiment analysis
# I am going to be using the 50% of our data and a representative sampling of 100 rows
sent_df <- as_tibble(training[1:100, c("text", "label")])
sent_tokens <- unnest_tokens(sent_df, w_tokens, text, token = "words", to_lower = TRUE) 
sent_tokens <- sent_tokens %>% anti_join(stop_words, by = c("w_tokens" = "word"))

# Top 10 words of author
top_words <- sent_tokens %>% group_by(w_tokens, label) %>% 
  tally(sort = T,  name = "freq")
top10_l1 <- sent_tokens %>% group_by(w_tokens) %>% 
  filter(label == 1) %>% 
  tally(sort = T,  name = "freq") %>% top_n(10, freq)
top10_l2 <- sent_tokens %>% group_by(w_tokens) %>% 
  filter(label == 2) %>% 
  tally(sort = T,  name = "freq") %>% top_n(10, freq)

# Arthur Conan Doyle top 10 words
ggplot(top10_l1, aes(x = w_tokens, y = freq)) + theme_bw() +
  geom_bar(stat="Identity") + labs(y = "Frequency", x = "Words", title = "Arthur Conan Doyle's top 10 words")
ggsave("plots/plot2.png", plot = last_plot())

# Charles Darwin top 10 words
ggplot(top10_l2, aes(x = w_tokens, y = freq)) +
  geom_segment(aes(x = w_tokens, xend = w_tokens, y = 1, yend = freq), size = 0.6) +
  geom_point(size = 4, color="cyan4", shape=10, stroke=1) + theme_bw() +
  theme(panel.grid.major.x = element_blank(), panel.border = element_blank(), axis.ticks.x = element_blank()) +
  labs(y = "Frequency", x = "Words", title = "Charles Darwin's top 10 words")
ggsave("plots/plot3.png", plot = last_plot())

# from the top 100 words; which words do the authors share in common?
ggplot(top_words[1:50,], aes(x = w_tokens, y = freq, fill = label)) +
  geom_bar(stat="Identity") + coord_flip() + theme_bw() +
  labs(y = "Frequency", x = "Words", title="Top shared common words", caption = "Data from UCI Machine learning Repository") +  
  scale_fill_discrete(name = "Authors", labels = c("Arthur Conan Doyle", "Charles Darwin"))
ggsave("plots/plot4.png", plot = last_plot())

#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Sentiment Analysis
positive <- get_sentiments(lexicon = "bing") %>% 
  filter(sentiment == "positive")

# positive sentiment analysis for Arthur Conan Doyle
top_words %>% 
  filter(label == 1) %>% 
  semi_join(positive, by = c("w_tokens" = "word"))

# positive sentiment analysis for charles darwin
top_words %>% 
  filter(label == 2) %>% 
  semi_join(positive, by = c("w_tokens" = "word"))

# sentiment score for words
bing <- get_sentiments(lexicon = "bing")
bing <- tibble::rowid_to_column(bing, "id")
bing_s <- sent_tokens %>% 
  group_by(label) %>% 
  inner_join(bing, by = c("w_tokens" = "word")) %>% 
  count(w_tokens,  sentiment) %>% 
  spread(sentiment, n, fill = 0) %>%  
  mutate(sentiment = positive - negative)
bing_s <- tibble::rowid_to_column(bing_s, "id")

# negative and positive sentiment of the first 50 words: using the bing dictionary
set.seed(12456)
bing_s <- shuffleRows(bing_s)
ggplot(bing_s[1:50,], aes(x = w_tokens, y = sentiment, fill = label)) + coord_flip() + geom_col(position = "stack", show.legend = F) + theme_bw() +
  labs(y = "Frequency", x = "Word", title="Sentiments: Bing Dictionary", subtitle = "Negative and Positive Words", caption = "Data from UCI Machine learning Repository") 
ggsave("plots/plot5.png")

# negative and positive sentiments analysis of authors words: using the bing dictionary
ggplot(bing_s, aes(x = id, y = sentiment, fill = label)) + geom_col() + theme_bw() +
  labs(y = "Sentiment", x = "Frequency", title="Negative and Positive Sentiments", subtitle = "Bing Dictionary", caption = "Data from UCI Machine learning Repository") + 
  scale_fill_discrete(name = "Authors", labels = c("Arthur Conan Doyle", "Charles Darwin"))
ggsave("plots/plot6.png")

# Comparing sentiment dictionaries for differences in categorization of snetiments
afinn <- get_sentiments(lexicon = "afinn")
afinn <- sent_tokens %>% 
  inner_join(afinn, by = c("w_tokens" = "word")) %>% 
  mutate(method = "afinn") %>% 
  rename(sentiment = value) 
afinn <- tibble::rowid_to_column(afinn, "id")


# nrc dictionary 
nrc <- get_sentiments(lexicon = "nrc")
nrc <- tibble::rowid_to_column(nrc, "id")

# bing and nrc dictionaries combined
bing_and_nrc <- bind_rows(sent_tokens %>% 
                            group_by(label) %>% 
                            inner_join(bing, by = c("w_tokens" = "word")) %>% 
                            mutate(method = "bing"), sent_tokens %>% 
                             group_by(label) %>% 
                            inner_join(nrc, by = c("w_tokens" = "word")) %>% 
                            filter(sentiment %in% c("positive", "negative")) %>% 
                            mutate(method = "nrc")) %>% 
  count(method, id, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>%  
  mutate(sentiment = positive - negative) 


bing_and_nrc <- bing_and_nrc %>% select(-id)
bing_and_nrc <- tibble::rowid_to_column(bing_and_nrc, "id")
bing_and_nrc <- shuffleRows(bing_and_nrc) 

ggplot(bing_and_nrc, aes(x = id, y = sentiment, fill = label)) + geom_col() + theme_bw() +
  labs(y = "Sentiment", x = "Frequency", title="Comparing Sentiment Dictionaries", 
       subtitle = "Negative and Positive Sentiments", caption = "Data from UCI Machine learning Repository") + 
  scale_fill_discrete(name = "Authors", labels = c("Arthur Conan Doyle", "Charles Darwin")) + facet_wrap(~method, ncol = 1, scales = "free_y")
ggsave("plots/plot7.png")

# nrc 
nrc_counts <- sent_tokens %>% 
  inner_join(nrc, by = c("w_tokens" = "word")) %>% 
  count(w_tokens, sentiment, sort = T) %>% 
  ungroup()

# nrc sentiment categories
ggplot(nrc_counts[1:100,], aes(x = w_tokens, y = n, fill = sentiment))  + geom_col(show.legend = F) + theme_bw() +
  labs(y = "Sentiment", x = "Frequency", title="NRC categories", 
       subtitle = "Sentiments: anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise and trust",
       caption = "Data from UCI Machine learning Repository") + facet_wrap(~sentiment, scales = "free_y") + coord_flip()
ggsave("plots/plot8.png")

# word cloud plot 1 
sent_tokens %>% 
  count(w_tokens) %>% 
  with(wordcloud(w_tokens, n, max.words = 100, random.order=TRUE, random.color = TRUE, rot.per=0.1, scale = c(3.0,0.5), colors = brewer.pal(11, "Spectral")))


# Analysing shingles
# ngram = 2; seperate filter and combine individual stopwords 
ngram_2 <- function(sent_df, shingles, text, shingle1, stop_words, shingle2, num, label) {
  ngram <- unnest_tokens(sent_df, shingles, text, token = "ngrams", n = num) 
  bi_sep <<- ngram %>% 
    separate(shingles, c("shingle1", "shingle2"), sep = " ", remove = T, convert = F)
  bi_fil <- bi_sep %>% 
    filter(!shingle1 %in% stop_words$word) %>% 
    filter(!shingle2 %in% stop_words$word) 
  bi_count <- bi_fil %>% group_by(label) %>% count(shingle1, shingle2, sort = T, name = "total") 
  bigram <- bi_fil %>% unite(shingles, shingle1, shingle2, sep = " ", remove = T)
  bigram %>% group_by(label) %>% count(shingles, sort = T, name = "freq")
}

ngram_2(sent_df, shingles, text, shingle1, stop_words, shingle2, num = 2, label)

# ngrams = 3. # not alot going here.
ngram_3 <- function(sent_df, shingles, text, shingle1, stop_words, shingle2, shingle3, shingle4, num, label) {
  ngram <- unnest_tokens(sent_df, shingles, text, token = "ngrams", n = num) 
  sep <- ngram %>% 
    separate(shingles, c("shingle1", "shingle2", "shingle3", "shingle4"), sep = " ", remove = T, convert = F)
  fil <- sep %>% 
    filter(!shingle1 %in% stop_words$word) %>% 
    filter(!shingle2 %in% stop_words$word) %>% 
    filter(!shingle3 %in% stop_words$word) %>% 
    filter(!shingle4 %in% stop_words$word)
  ngram <- fil %>% unite(shingles, shingle1, shingle2, shingle3, shingle4, sep = " ", remove = T, na.rm = T)
  ngram %>% group_by(label) %>% count(shingles, sort = T, name = "freq")
}

ngram_3(sent_df, shingles, text, shingle1, stop_words, shingle2, shingle3, shingle4, num = 3, label)

# remeber from our previous sentiment anlysis, the word "mother" seem to have a huge importance in positive and negative sentiments 
# lets see what that is all about in a bigram settings 
bi_sep %>%
  filter(shingle2 == "mother") %>%
  count(shingle1, shingle2, sort = TRUE)

s_girl <- bi_sep %>%
  filter(shingle2 == "girl") %>%
  inner_join(nrc, by = c("shingle1" = "word")) %>%
  count(shingle1, shingle2, sentiment, sort = TRUE)

s_girl

#--------------------------------------------------------------------------------------------------------------------------------------------------------
# igraph df
# using the full dataset
igraph <- function(df, shingles, text, shingle1, stop_words, shingle2, label) {
  igraph_df <- as_tibble(df, c("text", "label"))
  ngram <- unnest_tokens(igraph_df, shingles, text, token = "ngrams", n = 2) 
  bi_sep <- ngram %>% 
    separate(shingles, c("shingle1", "shingle2"), sep = " ", remove = T, convert = F)
  bi_fil <- bi_sep %>% 
    filter(!shingle1 %in% stop_words$word) %>% 
    filter(!shingle2 %in% stop_words$word)  
  igraph_c <- bi_fil %>% group_by(label) %>% 
    count(shingle1, shingle2, sort = T, name = "total") 
}
igraph_count <- igraph(df, shingles, text, shingle1, stop_words, shingle2, label)

# top combinations: igrpah counts + data frame to plot 
igraph_pp <- function(igraph_count, total) {
  igraph_pp <- igraph_count %>%
    filter(total > 10) %>%
    graph_from_data_frame()
}
igraph_pp <- igraph_pp(igraph_count, total)

# network graph
net_g <- function(grid, arrow, igraph_pp, total, name) {
  set.seed(0988)
  ar <- grid::arrow(type = "closed", length = unit(.15, "inches"))
  ggraph(igraph_pp, layout = "fr") +
    geom_edge_link(aes(edge_alpha = total, edge_width = total), show.legend = FALSE, arrow = ar, edge_colour = "cyan4", end_cap = circle(.07, 'inches')) +
    geom_node_point(color = "red", size = 3) +
    geom_node_text(aes(label = name), repel = T, point.padding = unit(0.2, "lines")) +
    theme_void() + ggtitle("Word Network")
}

net_g(grid, arrow, igraph_pp, total, name)
