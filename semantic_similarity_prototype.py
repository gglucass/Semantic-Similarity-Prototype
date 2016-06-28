import spacy
import re
import numpy
import pandas
import sense2vec
import math
from math import pi, sin, cos
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering
from itertools import combinations
nlp = spacy.load('en')
# Download glove word2vec from http://nlp.stanford.edu/projects/glove/
model = Word2Vec.load('glove.6B.300d_word2vec')

path = "PATH_TO_FOLDER"

neurohub_relationships = eval(open(path + 'neurohub_relationships', 'r').readlines()[0])
neurohub_concepts = eval(open(path + 'neurohub_concepts', 'r').readlines()[0])

companies = {'neurohub':[neurohub_concepts,neurohub_relationships]}

def plot_company(companies):
  for c in companies:
    list_of_concepts = companies[c][0]
    list_of_concepts = [l.lower() for l in list_of_concepts]
    relationships = companies[c][1]
    #create a cluster
    clusters, labels = plot_concepts(list_of_concepts, relationships, c)


def plot_all(companies):
  for c in companies:
    list_of_concepts = companies[c][0]
    list_of_concepts = [l.lower() for l in list_of_concepts]
    relationships = companies[c][1]
    #create a cluster
    clusters, labels = plot_concepts(list_of_concepts, relationships, c)
  
  for cl in clusters:
    k = math.floor(math.sqrt(len(clusters[cl])))
    name = c + "_" +labels[cl][0]
    name = name.replace('/', ':')
    subcluster, sublabels = plot_concepts(clusters[cl], relationships, name, k)
    
    cluster_combs = [comb for comb in combinations(clusters.keys(), 2)]
    for cc in cluster_combs:
      cluster_subdict = {x: clusters[x] for x in clusters if x in cc}
      keys = list(cluster_subdict.keys())
      name = c + "_rel" + ''.join('_' + str(k) for k in keys)
      cluster_0 = clusters[keys[0]]; cluster_1 = clusters[keys[1]]
      sublabels_0, subclusters_0 = create_clusters(cluster_0, clusters_to_make=math.floor(math.sqrt(len(cluster_0))))
      sublabels_0 = build_abstracted_terms_dictionary(subclusters_0)
      sublabels_1, subclusters_1 = create_clusters(cluster_1, clusters_to_make=math.floor(math.sqrt(len(cluster_1))))
      sublabels_1 = build_abstracted_terms_dictionary(subclusters_1)
      icr_z = find_intercluster_relationships_zoom(subclusters_0, subclusters_1, relationships)
      plot_relationship_zoom(cluster_subdict, keys, labels, subclusters_0, sublabels_0, subclusters_1, sublabels_1, icr_z, name)

def plot_relationship_zoom(clusters, keys, labels, subclusters_0, sublabels_0, subclusters_1, sublabels_1, ic_relationships, name):
  plt.xlim(0.0, 4)
  plt.ylim(0.0, 4)
  loc_0 = 1
  loc_1 = 3
  #left cluster
  size = len(clusters[keys[0]])
  plt.axis('off')
  plt.plot(1,1, 'o', markerfacecolor="None", markeredgecolor='k', markersize=70)
  plt.annotate(labels[keys[0]][0], xy=(loc_0,0.35), va="bottom", ha='center', size=8)
  #right cluster
  size = len(clusters[keys[1]])
  plt.axis('off')
  plt.plot(3,1, 'o', markerfacecolor="None", markeredgecolor='k', markersize=70)
  plt.annotate(labels[keys[1]][0], xy=(loc_1,0.35), va="bottom", ha='center', size=8)
  draw_zoom_relationships(ic_relationships, subclusters_0, subclusters_1, loc_0, loc_1)
  draw_zoom_cluster(subclusters_0, sublabels_0, loc_0)
  draw_zoom_cluster(subclusters_1, sublabels_1, loc_1)
  plt.savefig('/Users/Garm/Desktop/images/%s.pdf' % name, bbox_inches='tight')
  plt.clf()

def draw_zoom_cluster(subcluster, sublabels, location):
  num_clusters = len(subcluster)
  colors = plt.cm.Spectral(numpy.linspace(0.25, 0.75, num_clusters))
  grid_size = 1
  location = location-1
  r = 0.5
  for k, col in zip(set(sublabels),colors):
    x, y = calc_xy_circle(k, grid_size, num_clusters)
    plt.plot(x+r+location, y+r, 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)
    plt.annotate(sublabels[k][0], xy=(x+r+location,y+r), va='center', ha='center', size=3, fontweight='bold')

def draw_zoom_relationships(relationships, subclusters_0, subclusters_1, loc_0, loc_1):
  r = 0.5
  loc_0 = loc_0-1
  loc_1 = loc_1-1
  for rel in relationships:
    x1, y1 = calc_xy_circle(rel[0][1], 1, len(eval("subclusters_%s" % rel[0][0])))
    x2, y2 = calc_xy_circle(rel[1][1], 1, len(eval("subclusters_%s" % rel[1][0])))
    x1 = x1 + eval("loc_%s" % rel[0][0])+r
    x2 = x2 + eval("loc_%s" % rel[1][0])+r
    y1 = y1 + r; y2 = y2 + r
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=1)
    # if rel[0][0] != rel[1][0]:
    #   plt.annotate(''.join(x + ' ' for x in rel[2]), xy=((x1+x2)/2, (y1+y2)/2), size=3)




def find_intercluster_relationships_zoom(clusters_0, clusters_1, relationships):
  intracluster_relationships = []
  for relationship in relationships:
    pos1 = find_term_loc_in_clusters(relationship[0], clusters_0, clusters_1)
    pos2 = find_term_loc_in_clusters(relationship[2], clusters_0, clusters_1)
    if pos1 != pos2 and pos1 is not None and pos2 is not None:
      ls = [pos1, pos2, relationship]
      intracluster_relationships.append(ls)
  return intracluster_relationships

def find_term_loc_in_clusters(term, clusters_0, clusters_1):
  for c in clusters_0:
    if term.lower() in clusters_0[c]:
      return 0, c
  for c in clusters_1:
    if term.lower() in clusters_1[c]:
      return 1, c




def plot_concepts(list_of_concepts, relationships, name, clusters_to_make=9):
    labels, clusters = create_clusters(list_of_concepts, clusters_to_make=clusters_to_make)
    abstracted_terms = build_abstracted_terms_dictionary(clusters)
    number_of_labels = len(set(abstracted_terms))
    rels = count_intracluster_relationship(clusters, relationships)
    draw_relationships(clusters, rels, number_of_labels)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    draw_clusters(clusters, abstracted_terms, number_of_labels)
    plt.axis('off')
    plt.savefig('/Users/Garm/Desktop/images/%s.pdf' % name, bbox_inches='tight')
    plt.tight_layout()
    plt.clf()
    return clusters, abstracted_terms

def draw_clus(clusters, abstracted_terms,  number_of_labels):
  draw_clusters(clusters, abstracted_terms, number_of_labels)
  plt.axis('off')
  plt.savefig('/Users/Garm/Desktop/%s.pdf' % name, bbox_inches='tight')
  plt.clf()

def draw_relationships(clusters, rels, number_of_labels):
  grid_size = math.ceil(math.sqrt(number_of_labels))
  for rel in rels:
    x1, y1 = calc_xy_circle(rel[0], grid_size, number_of_labels)
    x2, y2 = calc_xy_circle(rel[1], grid_size, number_of_labels)
    width = rels[rel]
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=1+math.sqrt(width))

def draw_clusters(clusters, labels, number_of_labels):
  colors = plt.cm.Spectral(numpy.linspace(0.25, 0.75, number_of_labels))
  grid_size = math.ceil(math.sqrt(number_of_labels))
  plt.xlim(0.05, grid_size+0.05)
  plt.ylim(0.05, grid_size+0.05)
  for k, col in zip(set(labels),colors):
    x, y = calc_xy_circle(k, grid_size, number_of_labels)
    plt.plot(x, y, 'o', markerfacecolor=col, markeredgecolor='k', markersize=30+math.sqrt(len(clusters[k]))*13)
    plt.annotate(labels[k][0], xy=(x,y), va='center', ha='center', size=14, fontweight='bold')

def calc_xy_circle(i, grid_size, number_of_labels):
  if number_of_labels <3: number_of_labels = 3
  cx = cy = grid_size/2
  r = grid_size/2.8
  rad = 2 * pi / number_of_labels * i
  x = cx + r * cos(rad)
  y = cy + r * sin(rad)
  return x, y

def calc_xy(i, grid_size):
  x = i%grid_size
  y = math.floor(i/grid_size)
  return x+(grid_size/10),y+(grid_size/10)

def calc_x_y(i, number_of_labels):
  k = number_of_labels + i
  number_of_labels = number_of_labels * 2
  if i%4 == 0:
    x = number_of_labels - k
    y = number_of_labels - k
  if i%3 == 0:
    x = k
    y = number_of_labels - k
  if i%2 == 0:
    x = number_of_labels - k
    y = k
  else:
    x = k
    y = k
  if x == 0: x +=1
  if y == 0: y +=1
  return x, y

def create_clusters(list_of_words, clusters_to_make=9):
  df, labels_array = build_word_matrices(list_of_words, kind='twod')
  linkage = 'ward' 
  ac = AgglomerativeClustering(linkage=linkage, n_clusters=clusters_to_make)
  ac.fit(df)
  cluster_labels    = ac.labels_
  cluster_to_words  = find_word_clusters(labels_array, cluster_labels)
  print_clusters(cluster_to_words)
  return cluster_labels, cluster_to_words

  # birch = Birch(n_clusters=None, threshold=1.05)
  # birch.fit(df)
  # cluster_labels    = birch.labels_
  # cluster_to_words  = find_word_clusters(labels_array, cluster_labels)
  # print_clusters(cluster_to_words)

  # kmeans_model      = KMeans(init='k-means++', n_clusters=clusters_to_make, n_init=10, max_iter=10000)
  # kmeans_model.fit(df)
  # cluster_labels    = kmeans_model.labels_
  # cluster_inertia   = kmeans_model.inertia_
  # cluster_to_words  = find_word_clusters(labels_array, cluster_labels)
  # print_clusters(cluster_to_words)



def build_abstracted_terms_dictionary(clusters):
  abstracted_terms = autovivify_list()
  for cls in clusters:
    if clusters[cls]:
      print(cls)
      abstracted_terms[cls].append(find_cluster_abstracted_term(clusters[cls]))
    else:
      abstracted_terms[cls].append(' ')
  return abstracted_terms

def find_cluster_abstracted_term(cluster):
  gensim_words = split_gensim_words(cluster)
  gensim_words = [w for w in gensim_words if not nlp.is_stop(w)]
  gensim_vector = sum(model[w] for w in gensim_words)
  if gensim_words is not 0 and  gensim_vector is not 0:
    most_similar = model.most_similar([gensim_vector])
    sim = [word[0] for word in most_similar if word[0] in gensim_words]
    if not sim:
      return get_closest_word_slow(gensim_vector, gensim_words)[1]
    else: 
      return sim[0]
  else:
    return ' '

def get_closest_word_slow(vector, words):
  cosine_similarity = [0, 'n/a']
  for word in words:
    new_cos_sim = numpy.dot(vector, model[word])/(numpy.linalg.norm(gensim_vector)* numpy.linalg.norm(model[word]))
    if new_cos_sim > cosine_similarity[0]:
      cosine_similarity = [new_cos_sim, word]
  return cosine_similarity

def split_gensim_words(cluster):
  new_words = []
  for w in cluster:    
    try:
      if model[w].any():
        new_words.append(w)
    except:
      new_words
      words = w.split(' ')
      for word in words:
        try:
          if model[word].any():
            new_words.append(word)
        except:
          new_words
  return new_words

def build_word_matrices(list_of_words, kind="twod"):
  if kind == "twod":
    return build_twod_word_matrices(list_of_words)
  elif kind == "vector":
    return build_word_vector_matrices(list_of_words)

# requires numpy, pandas, spacy as nlp
def build_twod_word_matrices(list_of_words):
  # list_of_words = preprocess_words(list_of_words)
  labels = [a for a in list_of_words]
  matrix_length = len(list_of_words)
  numpy_matrix = numpy.zeros((matrix_length,matrix_length))
  pandas_matrix = pandas.DataFrame(numpy_matrix, index=labels, columns=labels)
  for row_idx, (row_name, row) in enumerate(pandas_matrix.iterrows()):
    for col_idx, col in enumerate(pandas_matrix.columns):
      if pandas_matrix.iloc[row_idx][col_idx] == 0:
        word1 = nlp(row.name)
        word2 = nlp(col)
        pandas_matrix.iloc[row_idx][col_idx] = pandas_matrix.iloc[col_idx][row_idx] = word1.similarity(word2)
  return pandas_matrix, labels

def build_word_vector_matrices(list_of_words):
  # list_of_words = preprocess_words(list_of_words)
  numpy_arrays = []
  labels_array = []
  for word in list_of_words:
    labels_array.append(word)
    numpy_arrays.append( numpy.array([float(i) for i in nlp.vocab[word].vector])  )
  return numpy.array( numpy_arrays ), labels_array

def build_word_vector_matrices_sense2vec(list_of_words):
  # list_of_words = preprocess_words(list_of_words)
  numpy_arrays = []
  labels_array = []
  for word in list_of_words:
    labels_array.append(word)
    numpy_arrays.append( numpy.array([float(i) for i in get_sense2vec_model(word, "NOUN")])  )
  return numpy.array( numpy_arrays ), labels_array

#Support stuff
def convert(name):
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

def preprocess_words(list_of_words):
  new_list_of_words = []
  for word in list_of_words:
    new_list_of_words.append(convert(word))
  return new_list_of_words

def find_word_clusters(labels_array, cluster_labels):
  cluster_to_words = autovivify_list()
  for c, i in enumerate(cluster_labels):
    cluster_to_words[ i ].append( labels_array[c] )
  return cluster_to_words

def print_clusters(clusters):
  for c in clusters:
    print(str(c) + ": " + str(clusters[c]))

def get_gensim_model(word):
  try:
    return model[word]
  except:
    return numpy.zeros(300, dtype='float32')

def get_sense2vec_model(word, word_type):
  try:
    return sense2vec_model["%s|%s" % (word, word_type)][1]
  except:
    return numpy.zeros(128, dtype='float32')

class autovivify_list(dict):
  def __missing__(self, key):
    value = self[key] = []
    return value
  def __add__(self, x):
    if not self and isinstance(x, Number):
      return x
    raise ValueError
  def __sub__(self, x):
    if not self and isinstance(x, Number):
      return -1 * x
    raise ValueError

def count_intracluster_relationship(clusters, relationships):
  ir = find_intracluster_relationships(clusters, relationships)
  ir_unique = {tuple(x):ir.count(x) for x in ir}
  return ir_unique


def find_intracluster_relationships(clusters, relationships):
  intracluster_relationships = []
  for relationship in relationships:
    pos1 = find_term_loc_in_cluster(relationship[0], clusters)
    pos2 = find_term_loc_in_cluster(relationship[2], clusters)
    if pos1 != pos2 and pos1 is not None and pos2 is not None:
      ls = [pos1, pos2]
      ls.sort()
      intracluster_relationships.append(ls)
  return intracluster_relationships

def find_term_loc_in_cluster(term, clusters):
  for c in clusters:
    if term.lower() in clusters[c]:
      return c