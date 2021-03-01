import xml.etree.ElementTree as ET
import os
import sys
sys.path.append("conceptual_spaces")
import cs.cs as space
from cs.weights import Weights
from cs.cuboid import Cuboid
from cs.core import Core
from cs.core import from_cuboids
from cs.concept import Concept
import matplotlib.pyplot as plt
import visualization.concept_inspector as ci
import numpy as np
from matplotlib import patches
from sklearn.metrics import average_precision_score
from functools import reduce
from math import isnan
import random
import _pickle as pkl

C=0.05
#Get data from path to xml and add to given data
def xml_to_dict(path, data = {}, domain_mapping={}, dimension_names=[]):
    with open(os.path.normpath(path), 'rb') as xml_file:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        newdata={}
        for exemplar in root:
            new_concept = get_domains(exemplar)
            if(new_concept):
                newdata[exemplar.attrib['name']] = new_concept

    n_old=len(dimension_names)
    domains={}
    for exemplar in newdata.values():
        domains.update(exemplar)
    del(domains['supercategory'])

    #domains = mapping: domain->dimension indices, also add new dimension names
    if domain_mapping:
        i=sum([len(domain) for domain in domain_mapping.values()])
    else:
        i=0
    for key in domains:
        if not key in domain_mapping:
            for dim in domains[key]:
                dimension_names+=[dim]
            domain_mapping[key]=list(range(i,i+len(domains[key])))
            i+=len(domains[key])

    #kick inconsistents
    to_del=[]
    for datum in newdata:
        if not is_consistent(newdata[datum], domain_mapping):
            print(datum ,'is inconsistent')
            to_del+=[datum]
    for kickit in to_del:
        del(newdata[kickit])

    #translate to point
    for datum in newdata:
        point=[]
        for domain in domain_mapping:
            if domain in newdata[datum]:
                point+=list(newdata[datum][domain].values())
            else:
                point+=[float('inf')]*len(domain_mapping[domain])     
        newdata[datum]={'supercategory':newdata[datum]['supercategory'], 'point':point}
    if len(dimension_names)>n_old and n_old:
        for datum in data:
            data[datum]['point']=data[datum]['point']+[float('inf') for _ in range(len(dimension_names)-n_old)]
    data.update(newdata)
    return domain_mapping, data, dimension_names

def less_dimensions(domain_mapping, data, dim_names,threshold=10):
    dimcounter={}
    for domain in domain_mapping:
        dimcounter[domain]=sum([1 if not datum['point'][domain_mapping[domain][0]] == float('inf') else 0 for datum in data.values()])


    for domain in dimcounter:
        if dimcounter[domain]<threshold:
            del(domain_mapping[domain])
    indices=[value for domain in domain_mapping.values() for value in domain]
    for concept in data.values():
        concept['point']=[value for value,i in zip(concept['point'],range(len(concept['point']))) if i in indices]

    dim_names=[value for value,i in zip(dim_names,range(len(dim_names))) if i in indices]
    i=0
    for domain in domain_mapping:
        if domain_mapping[domain][0]==i:
            i+=len(domain_mapping[domain])
        else:
            domain_mapping[domain]=[i+j for j in range(len(domain_mapping[domain]))]
            i+=len(domain_mapping[domain])
    return domain_mapping, data, dim_names
    
def family_into_space(name, values, add=True):
    cuboids=[]
    domains = {}
    for point in values:
        subdomains=point_to_domains(point)
        cuboids.append(Cuboid([dim if not dim==float('inf') else -dim for dim in point], point, subdomains))
        domains.update(subdomains)

    core=from_cuboids(cuboids,domains)
    weights=Weights(space._def_dom_weights,space._def_dim_weights)
    concept=Concept(core, 1.0, C, weights)
    if add:
        space.add_concept(name,concept)
    return concept
         
#Group data by their supercategories to form concepts
def form_supercategories(data):
    concepts={}
    for datum in data.values():
        if datum['supercategory'] in concepts:
            concepts[datum['supercategory']].append(datum['point'])
        else:
            concepts[datum['supercategory']]=[datum['point']]
        #del(concepts[datum['supercategory']])
    return concepts
    
#helper for xml_to_dict, important for how the data is supposed to look like
def get_domains(xml_elem):
    domains = xml_elem.find('genericPhysicalDescription')
    if(not(domains)):
        return
    domain_dict={}
    try:
        supercategory = xml_elem.find('family').text
        if supercategory in ['feline','rodent','primate','cetacean']:
            supercategory='mammal'
        elif supercategory in ['crustacean','arachnid']:
            supercategory = 'arthropod'
        elif supercategory in ['fruit', 'transport', 'furniture', 'book', 'building', 'musical_instrument', 'present', 'architectural_element', 'amphibian']:
            supercategory = 'object'
            return
        domain_dict['supercategory']=supercategory
    except AttributeError:
        domain_dict['supercategory']="object"
        return
        
    for domain in domains:
        domain_name=domain.tag
        if domain_name=='hasPart':
            domain_name=domain.get('name')
            if domain[0].tag=='number':
                domain_dict['n_'+domain_name]={'number_'+domain_name:float(domain[0].text)}
            else:
                pass
        else:
            numerical_domain={}
            for subelem in domain:
                try:
                    numerical_domain[subelem.tag]=float(subelem.text)
                except ValueError:
                    pass
            if numerical_domain:            
                if domain_name=='size' or domain_name=='location' or domain_name=='color' or domain_name=='locomotion':
                    domain_dict[domain_name]={key:value for key,value in zip(numerical_domain.keys(),numerical_domain.values())}
    return domain_dict

def domains_from_point(point):
    domains={}
    for domain in space._domains:
        if all(not point[dim]==float('inf') for dim in space._domains[domain]):
            domains[domain]=space._domains[domain]
    return domains

def point_to_concept2(point, name, size=100000, weights=[]):
    domains = domains_from_point(point)
    p_min=[-size for value in point]
    p_max=[size for value in point]
    c_example = Cuboid(p_min, p_max, domains)
    s_example = Core([c_example], domains)
    if not weights:
        weights=space._def_dom_weights
    w_example = Weights(weights,space._def_dim_weights)
    concept = Concept(s_example, 1.0, C, w_example)
    space.add_concept(name, concept)
    return concept

def point_to_concept(point, name, weights=[]):
    domains = domains_from_point(point)
    p_min=[value if not value==float('inf') else float('-inf') for value in point]
    c_example = Cuboid(p_min, point, domains)
    s_example = Core([c_example], domains)
    if not weights:
        weights=space._def_dom_weights
    w_example = Weights(weights,space._def_dim_weights)
    concept = Concept(s_example, 1.0, C, w_example)
    space.add_concept(name, concept)
    return concept

#gives all permutations of a binary vector of given length with -1 and 1 as elements
def get_permutations(length):
    if length==1:
        #positive first, because later on earlier permutations are preferred
        return [[[1]],[[-1]]]
    else:
        new_vectors=[]
        for vector in get_permutations(length-1):
            new_vectors+=[vector+[[1]],vector+[[-1]]]
        return new_vectors
    
def evaluate_signs(signs):
    #max(evaluate_signs) -> maximize cumulative similarity of columns
    value=0
    for dimension in signs.T:
        value+=abs(sum(dimension))
    return value

#'brushes' all vectors to point in the most common general direction
def brush_vectors(pc1s):
    signs=np.zeros(pc1s.shape)
    for j in range(len(signs)):
        for i in range(len(signs[j])):
            if pc1s[j,i]>0:
                signs[j,i]=1
            elif pc1s[j,i]<0:
                signs[j,i]=-1
    permutations=get_permutations(len(signs))
    signs_x_perms=np.array([perm*signs for perm in permutations])
    return pc1s*permutations[np.argmax([evaluate_signs(brushed_pc1s) for brushed_pc1s in signs_x_perms])]

def brush_to_size(pc1s):
    indices=[space._dim_names.index('x'),space._dim_names.index('y'),space._dim_names.index('z')]
    newpc1s=[]
    for pc1 in pc1s:
        if sum(pc1[indices])>0:
            newpc1s.append(pc1)
        else:
            newpc1s.append(-pc1)
    return np.array(newpc1s)


#We do not want any concepts with partially filled domains. This would go against the very idea of domains. input: {doms:values}. Also concepts with only undefined dims are not needed
def is_consistent(concept, domain_mapping={}):
    inf_check=[]
    '''if not domain_mapping:
        domain_mapping=space._domains'''
    for domain in concept:
        if not domain=='supercategory':
            if not(len(concept[domain])==len(domain_mapping[domain])):
                return False
            inf_check+=concept[domain]
    if all([check==float('inf') for check in inf_check]):
        return False
    return True

#normalize by data with missing values replaced my mean
def normalize_to_standard_score(points):
    #calculate means af defined values for each dimension
    means=[]
    for dimension in zip(*points):
        values=[x for x in dimension if not (x==float('inf') or np.isnan(x))]
        if len(values):
            means.append(sum(values)/len(values))
        else:
            means.append(0)
    means=np.array(means)
    
    #copy data, replace missing values by mean
    normalized_data=np.zeros(np.shape(points))
    for j in range(len(points)):
        for i in range(len(points[j])):
            normalized_data[j][i]=points[j][i] if not (points[j][i]==float('inf') or np.isnan(points[j][i])) else means[i]
            
    #calculate the std in each dimension
    stds=np.std(normalized_data, axis = 0)
    #stds of 0 zero are considered to be inf, in order to avoid division by zero
    stds=np.array([std if not std==0 else float('inf') for std in stds])
    #normalize data to standard score
    normalized_data = [(point-means)/stds for point in normalized_data]
    #for the inverse function a std of zero is ok again, inverse fun not implemented, only "amount" of one std
    stds=np.array([std if not std==float('inf') else 0 for std in stds])
    #inverse_fun = lambda x:x*stds+means
    return stds, np.array(normalized_data)

'''def get_names(domains):
    dimension_names = []
    for domain in domains.values():
        dimension_names += [dim for dim in domain.keys()]'''
        
'''def dict_to_point(concept_dict):
    point=[concept_dict.get(domain,{str(key):float("inf") for key in range(len(space._domains[domain]))}).values() for domain in space._domains]
    #flatten
    point= [p for dom in point for p in dom]
    return point'''

def point_to_domains(point):
    domains={}
    for dim in range(space._n_dim):
        if not(point[dim]==float('inf') or point[dim]==float('-inf')):
            domains[space.dim_to_dom[str(dim)]]=space._domains[space.dim_to_dom[str(dim)]]
    return domains

def one_shot(point, method=None, sibblings=None, name="newlearned", learn_weights=False):
    #right now unnecessary because by now only two_cubs_PCA is left over (actually naive_with_size, too), but cool python stuff
    return method(point, sibblings, learn_weights=learn_weights)

def two_cubs_PCA(point, sibblings, learn_weights=False):
    #sibbling is list of sibbling names or dict with sibbling names as keys
    #We can play around with artificial_scaling to see the effect of arbitrary generalization (<1 does not really mean more specific, but rather just smaller/conservative)
    artificial_scaling=1
    n_dims=len(point)
    if sibblings==None:
        #not implemented for some reason
        sibblings=get_sibblings(supercategory)   
    variances=[]
    pc1s=[]
    #correlations=[]
    examples_x_stds=[]#Matrix examples x stds
    avg = np.average#lambda values: sum(values)/len(values)
    allcenters=[]
    for sibbling in sibblings:
        #centers of all cuboids of the sibbling concept
        centers = np.array([[(p_max+p_min)/2 for p_max,p_min in zip(cuboid._p_max, cuboid._p_min)] for cuboid in space._concepts[sibbling]._core._cuboids])
        #throw away concepts with less than 2 cuboids (not happening with dataset)
        if len(centers)>1:
            allcenters.append(centers)
            #Centers are translated to data cloud that has std=1 in every dimension. stds remembers the multiplicative(eng?) part of the reverse operation
            #as simpler version compared to regression
            stds, data=normalize_to_standard_score(centers)
            examples_x_stds.append(stds)
            #np.cov input: rows=dimensions, columns=observations!! (very likely to produce error down the line if done wrong though)
            #np.linalg.eig output: values=eigenvalues, vectors=eigenvectors in columns!! (no error when done wrong because square matrix!)
            values, vectors = np.linalg.eig(np.cov(data.T))
            #the first PC has unit length so multiplying it with std lets it have (euclidean) length=std (in direction of largest var)
            pc1s.append(np.array(vectors[:,np.argmax(values)])*(max(values)**(1/2)))
        
    #The sign of a whole PC is only meaningfull in relation to the other PCs, so we assume that all first PCs point in a somewhat similar direction. Size is ignored.
    pc1s=brush_vectors(np.array(pc1s))
    #makes signs point in a direction so that the size dimension still correlate with each other over all examples
    pc1s=brush_to_size(pc1s)
    
    #A dimension of a PC that is zero because the original values where all undefined is considered uninformative, since examples
    #that have an undefined value lead to a concept with no variance in that dimension anyways
    meanpc=[]
    for dimension, dimindex in zip(pc1s.T, range(len(pc1s.T))):
        #all meaningfull values of the 1st PCs of all known concepts in a single dimension
        cleanvalues=[]
        for value, conceptindex in zip(dimension, range(len(dimension))):
            #print([np.isnan(examplevalue[dimindex]) for examplevalue in allcenters[conceptindex]])
            #print(conceptindex)
            if not value==0:
                cleanvalues.append(value)
            #a variance of 0 in a certain dimension in the first PC can be due to all known examples of a category being undefined in that dimension
            #the other way round this implies that some values have not been included because of the wrong reasons beforehand
            elif not all([np.isnan(examplesvalues[dimindex]) for examplesvalues in allcenters[conceptindex]]):
                cleanvalues.append(value)

        if len(cleanvalues):
            #quite some information is lost in this step (PCs canceling out each other because of wrong reasons), but we need to generalize
            meanpc.append(sum(cleanvalues)/len(cleanvalues))
        else:
            #Should not happen, since uninformative dimensions have already been kicked out. If it happens anyways, zero  is actually the correct value
            #(i.e. no variance in dim with only undefined values)
            meanpc.append(0)

    meanstds=np.average(examples_x_stds, axis=0)

    
    #for some reason i=0 happens quite often, although the docu promises that real values are used in these cases
    meanpc=[np.real(val) for val in meanpc]
    
    #We have our background knowledge. Nice! Multiplying with the mean 'size' in each dimension is improvable (multiplicative part of regression would be better)
    learnt_vector=meanstds*meanpc
    
    #start building our new concept
    cub1=np.zeros(n_dims)
    cub2=np.zeros(n_dims)
    
    for dim in range(n_dims):
        #point as central region, multiply by two to reverse taking center of cuboids
        cub1[dim]=point[dim]-learnt_vector[dim]*2*artificial_scaling
        cub2[dim]=point[dim]+learnt_vector[dim]*2*artificial_scaling

    c1_p_min=[cub1[dim] if cub1[dim] < point[dim] else float('-inf') if point[dim]==float('inf') else point[dim] for dim in range(n_dims)]
    c1_p_max=[cub1[dim] if cub1[dim] > point[dim] else point[dim] for dim in range(n_dims)]
    
    c2_p_min=[cub2[dim] if cub2[dim] < point[dim] else float('-inf') if point[dim]==float('inf') else point[dim] for dim in range(n_dims)]
    c2_p_max=[cub2[dim] if cub2[dim] > point[dim] else point[dim] for dim in range(n_dims)]
    
    domains = domains_from_point(c1_p_max)
    core=Core([Cuboid(c1_p_min, c1_p_max, domains),Cuboid(c2_p_min,c2_p_max,domains)],domains)
    
    if not learn_weights:
        dom_weights=space._def_dom_weights
    else:
        #dramatically improves performance, but seems unfair to compare to classes with meaningless weights
        dom_weights={}
        for domain in domain_mapping:
            variance=np.average([abs(meanpc[dim]) for dim in domain_mapping[domain]])+0.1            
            dom_weights[domain]=(1/variance)**100
        
    weights=Weights(dom_weights,space._def_dim_weights)
        
    concept=Concept(core, 1.0, C, weights)

    return concept
    
def naive_with_size(point, sibblings):
    #actually not bad for comparison, lets keep it for now
    learnt=two_cubs_pca(point,sibblings)
    size=learnt.size()
    print('size:',size)
    avg_dim_size=np.average([[np.average(cuboid._p_max[dimension]-cuboid._p_min[dimension]) for cuboid in learnt._core._cuboids] for dimension in range(len(learnt._core._cuboids[0]._p_min))])
    print(avg_dim_size)
    crisp_size=learnt.size_given_c(1000000000)
    print('crisp_size:',crisp_size)
    return avg_dim_size

def ap_balanced(concept, testdata={}, figure=None, target=None):
    class_size=len(testdata[target])-(len(testdata[target])%(len(testdata)-1))
    balanced_data={target:testdata[target][20:class_size+21]}
    for family in testdata:
        if not family==target:
            balanced_data[family]=testdata[family][:int(class_size/(len(testdata)-1))]
    return average_precision(concept,testdata=balanced_data, target=target)
    
def average_precision(concept, testdata={}, figure=None, target=None):#old signature!
    """Average Precision of the concept as a classifier

    Parameters:
    negatives (list): list of points that do not belong to the target class
    positives (list):...

    Returns:
    ap (float)

   """
    negatives=[point for family in testdata for point in testdata[family] if not family==target]
    positives=[point for family in testdata for point in testdata[family] if family==target]
    #negatives=reduce(lambda x,y:x+y, list(negatives.values()))
    y_true=np.concatenate((np.ones(len(positives)),np.zeros(len(negatives))))
    y_scores=np.zeros(len(y_true))
    for i in range(len(y_scores)):
        if i < len(positives):
            y_scores[i]=concept.membership_of(positives[i])
        else:
            y_scores[i]=concept.membership_of(negatives[i-len(positives)])
    sorted_truths=[[y,x] for y,x in sorted(zip(y_scores,y_true),reverse=True)]
    recalls=[]
    precisions=[]
    i=0
    while i < len(sorted_truths):
        j=0
        if i+1<len(sorted_truths):
            while sorted_truths[i][0]==sorted_truths[i+j+1][0]:
                if i+j+2<len(sorted_truths):
                    j+=1
                else:
                    break
        tp=sum([truth[1] for truth in sorted_truths[:i+1+j]])
        t=len(positives)
        p=i+j+1
        recalls.append(tp/t)
        precisions.append(tp/p)
        for _ in range(j):
            i+=1
        i+=1

    #plt.subplots(figsize=(10, 10))
    if figure=='part':
        plt.scatter(recalls, precisions, color = color)#[(0.5,x,x) for x in [(len(recalls)-i)/len(recalls) for i in range(len(recalls))]])
    elif figure=='developement':
        plt.scatter(recalls, precisions, color = [(0.5,x,x) for x in [(len(recalls)-i)/len(recalls) for i in range(len(recalls))]])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()
    #plt.figure(figsize=(10,20))
    #plt.show()
    ap=average_precision_score(y_true, y_scores)
    '''print(y_true)
    print(y_scores)
    print(ap)'''
    return ap

#normalizes values to sum(values)=1
def normalize(values):
    if not sum(values):
        return([1/len(values) for _ in values])
    summed=sum(values)
    for i in range(len(values)):
        values[i]=values[i]/summed

#cheap workaround, but lets be honest, that way we are on the save side, especially in jupyter
class Memorizer:
    memberships={}
    def memorize_memberships(learnt_classes, testdata):
        flattened_data=[datum for family in learnt_classes for datum in testdata[family]]
        for concept in learnt_classes:
            Memorizer.memberships[concept]=[learnt_classes[concept].membership_of(datum) for datum in flattened_data]
        print(Memorizer.memberships)
    
    def deserialize_memberships():
        with open('memberships', 'rb') as file:
            Memorizer.memberships=pkl.load(file)
    n_log0=0
    
def log(x):
    #in information theory log means log_2 since naturalis already has ln(x) as notation
    return np.log2(x)
     
def categorical_cross_entropy(concept, testdata, target, figure=None):
    '''compute the categorical cross entropy from a classifier consisting of our existing concepts, where one can be replaced
    with the concept parameter (concept for target), to a theoretical perfect classifier'''
    #Since officially there is no ordering in dictonaries, this:
    indices={family:index for family,index in zip(testdata,range(len(testdata)))}
    
    predictions=[]
    for family in indices:
        i=0
        for testdatum in testdata[family]:
            i+=1
            #absolute certainty of each class for the example
            all_predictions=[space._concepts[conc].membership_of(testdatum) if not conc==target else concept.membership_of(testdatum) for conc in indices]
            #relative certainty of each class for the example, with assumed equipropable a priori propabilities
            normalize(all_predictions)
            #only the prediction for the class from which the example was drawn is relevant, all others would be multiplied by zero down the line
            pred_target=all_predictions[indices[family]]
            predictions.append(pred_target)

    cross_entropies=[]
    #iterate through examples
    for pred in predictions:
        #might be inf
        cross_entropies.append(-log(pred))

    #annoying but really informative, regarding individual examples
    print('score:',np.median(cross_entropies))
    
    #median because mean is too susceptible for inf values
    return np.median(cross_entropies)

def sample_data(n, a=2):
    '''Crisp sampling with n examples drawn per class, n%10=0
    Returns: testdata (dict): {class: point (list)}
    '''
    domain_mapping, data, dim_names = xml_to_dict("Dataset\\prototypes.xml")
    domain_mapping, data, dim_names = xml_to_dict("Dataset\\exemplars.xml", domain_mapping=domain_mapping, data=data, dimension_names=dim_names)
    domain_mapping, data, dim_names = less_dimensions(domain_mapping, data, dim_names)
    space.init(len(dim_names), domain_mapping, dim_names)

    concepts=form_supercategories(data)

    random.seed(a=a)

    testdata={}
    families={}
    i=1
    for concept in concepts:
        print(concept)
        families[concept]=family_into_space(concept, concepts[concept],add=False)
        conc=families[concept]
        testdata[concept]=[]
        for j in range(int(n/10)):
            testdata[concept]+=conc.crisp_sample(10)
            print(i,j*10)
        i+=1
        
    #no examples without information
    for family in testdata:
        for i in range(n):
            while all([value==float('inf') for value in testdata[family][i]]):
                testdata[family][i]=families[family].crisp_sample(1)[0]
    
    return testdata
    
def serialize_testdata(testdata, path='testdata.pkl'):
    with open(path, 'wb') as file:
        pkl.dump(testdata,file)
        
def deserialize_testdata(path):
    with open(path, 'rb') as file:
        data=pkl.load(file)
    return data
    
def evaluate_example(example, data, concepts, domains, dim_names, figure=None, method=two_cubs_PCA, testdata={}, criterion=average_precision, learn_weights=False):
    """Look what would happen, if example was a OSL input

    Parameters:
    example (dict): {'target': name of target class (str), 'point': values of the datum (list), 'n': count of examples from target, used for naming (int)}
    or
    example (str): name of traindatum (e.g. 'Rhinozeros')

    concepts (dict): returned from form_supercategories, {class: traindata for class} 

    Returns:
    aps (dict): complete evaluation, ordered by inputs

    """
    space.init(len(dim_names), domains,dim_names)
    
    #translate traindata into concepts, and put them into space
    classes={name:family_into_space(name, values) for name,values in zip(concepts,concepts.values())}
    
    #criterion removed
    '''if criterion==complete_categorical_cross_entropy:
        Memorizer.deserialize_memberships()'''

    if isinstance(example,dict):
        target=example['target']
        examplename='some'+target+str(example['n'])
        example=example['point']
    else:
        #example taken from trainset (ok because of leave-one-out evaluation, but unbalanced)
        target=data[example]['supercategory']
        examplename=example
        example=data[example]['point']
        testdata=list(concepts[target])
        testdata.remove(data[examplename]['point'])

    '''testset=list(concepts[target])
    if not traindata:
        try:
            testset.remove(data[examplename]['point'])
        except:
            pass
    
    if not testdata:
        negatives={family:[point for point in trainset[family]] for family in trainset}
        positives=testset
    else:
        negatives=testdata.copy()
        del(negatives[target])
        positives=testdata[target]'''
    
    sibblings = [concept for concept in space._concepts if not concept==target]
    learnt_category=one_shot(example, method,sibblings=sibblings,name=target+'from'+examplename,learn_weights=learn_weights)
    
    #start cheating
    centers = np.array(concepts[target]).T
    means=[]
    for dimension in range(len(centers)):
        means.append(np.average([value for value in centers[dimension] if not value==float('inf')]))
    means=[mean if not np.isnan(mean) else float('inf') for mean in means]
    cheated=one_shot(means, method,sibblings=sibblings,name=target+'cheatedfrom'+examplename,learn_weights=learn_weights)
    for i in range(len(means)):
        if not means[i]==float('inf') and example[i]==float('inf'):
            means[i]=float('inf')
    cheated2=one_shot(means, method,sibblings=sibblings,name=target+'cheated2from'+examplename,learn_weights=learn_weights)
    target_dict={}
    known=space._concepts[target]
        
    if not learn_weights:
        weights=space._def_dom_weights
    else:
        weights=learnt_category._weights
        known._weights=weights

    #A size of 100000 entails all data we have in our data
    naive_big=point_to_concept2([0 for _ in example],'naive_big', 100000,weights)
    
    print('\n\nExample:',examplename)
    print('\nlearnt')
    eval_learnt=criterion(learnt_category, testdata=testdata, figure=figure, target=target)
    print('\nnaiveBig')
    ap_naive=criterion(naive_big, testdata=testdata, figure=figure, target=target)
    
    print('\nnaiveSmall')
    naive_small=point_to_concept(example,'naive',weights=weights)
    ap_naive2=criterion(naive_small, testdata=testdata,figure=figure, target=target)

    
    print('\nknown')
    ap_known=criterion(known, testdata=testdata, figure=figure, target=target)
    
    print('\ncheated:')
    ap_cheated=criterion(cheated, testdata=testdata, figure=figure, target=target)
    
    print('\ncheated2')
    ap_cheated2=criterion(cheated2, testdata=testdata, figure=figure, target=target)
    
    return {examplename:[eval_learnt, ap_naive, ap_naive2, ap_known, ap_cheated]}

def print_evaluation(aps):
    diff_ap=[x[0] for x in aps.values()]
    print('learnt')
    print('max',max(diff_ap),'(',list(aps.keys())[np.argmax(diff_ap)],')')
    print('min',min(diff_ap),'(',list(aps.keys())[np.argmin(diff_ap)],')')
    print('avg',sum(diff_ap)/len(diff_ap))

    diff_ap=[x[1] for x in aps.values()]
    print('naive big')
    print('max',max(diff_ap))
    print('min',min(diff_ap))
    print('avg',sum(diff_ap)/len(diff_ap))

    diff_ap=[x[2] for x in aps.values()]
    print('naive small')
    print('max',max(diff_ap),'(',list(aps.keys())[np.argmax(diff_ap)],')')
    print('min',min(diff_ap),'(',list(aps.keys())[np.argmin(diff_ap)],')')
    print('avg',sum(diff_ap)/len(diff_ap))

    diff_ap=[x[3] for x in aps.values()]
    print('known')
    print('max',max(diff_ap))
    print('min',min(diff_ap))
    print('avg',sum(diff_ap)/len(diff_ap))

    diff_ap=[x[4] for x in aps.values()]
    print('cheated')
    print('max',max(diff_ap))
    print('min',min(diff_ap))
    print('avg',sum(diff_ap)/len(diff_ap))
    
def evaluate_on_dataset(testdata, criterion):
    #read concepts from prototypes and exemplars
    domain_mapping, data, dim_names = xml_to_dict("Dataset\\prototypes.xml")
    domain_mapping, data, dim_names = xml_to_dict("Dataset\\exemplars.xml", domain_mapping=domain_mapping, data=data, dimension_names=dim_names)
    domain_mapping, data, dim_names = less_dimensions(domain_mapping, data, dim_names,threshold=10)
    space.init(len(dim_names), domain_mapping, dim_names)

    concepts=form_supercategories(data)

    aps={}
    for target in testdata:
        n=0
        if len(concepts[target])-1:
            for example in testdata[target]:

                n+=1
                ex={'target':target,'point':example,'n':n}
                #print(ex['target'],ex['n'])
                aps.update(evaluate_example(ex, data, concepts, domain_mapping, dim_names, method=two_cubs_PCA, testdata=testdata, criterion=criterion))
    return aps

'If you see this, the cell has succesfully finished running'

''

def generate(target='bird',dims=[6,7,8], n=None):
    #read concepts from prototypes and exemplars
    domain_mapping, data, dim_names = xml_to_dict("Dataset\\prototypes.xml")
    domain_mapping, data, dim_names = xml_to_dict("Dataset\\exemplars.xml", domain_mapping=domain_mapping, data=data, dimension_names=dim_names)
    domain_mapping, data, dim_names = less_dimensions(domain_mapping=domain_mapping, data=data, dim_names=dim_names)
    space.init(len(dim_names), domain_mapping, dim_names)
    concepts=form_supercategories(data)
    testdata=deserialize_testdata('testdata.pkl')
    if not n:
        n=random.choice(list(range(len(testdata[target]))))
    ex={'target':target,'point':testdata[target][n],'n':n}
    sibblings=[family for family in concepts if not family==target]
    print(ex)
    centers = np.array(concepts[target]).T
    means=[]
    for dimension in range(len(centers)):
        means.append(np.average([value for value in centers[dimension] if not value==float('inf')]))
    means=[mean if not np.isnan(mean) else float('inf') for mean in means]
                      
    aps=evaluate_example(ex, data, concepts, domain_mapping, dim_names, method=two_cubs_PCA, testdata=testdata, criterion=ap_balanced)
    aps=aps[list(aps.keys())[0]]
    space.init(len(dim_names), domain_mapping, dim_names)

    for name,values in zip(concepts,concepts.values()):
        family_into_space(name, values, add=True)
    space.add_concept('learnt_'+target,one_shot(ex['point'], method=two_cubs_PCA, sibblings=sibblings, name="learnt_"+target, learn_weights=False))
    space.add_concept('cheated_'+target,one_shot(means, method=two_cubs_PCA,sibblings=sibblings,name=target+'cheated_'+target,learn_weights=False))
        
    
    del(space._concepts['mammal'])
    

    ci.init(dims=dims)
    #ap_learnt, ap_naive, ap_known, ap_cheated, ap_supercheated
    print('learnt',aps[0])

    print('small',aps[1])

    print('big',aps[2])

    print('known',aps[3])

    print('cheated',aps[4])

def display_space():
    domain_mapping, data, dim_names = xml_to_dict("Dataset\\prototypes.xml")
    domain_mapping, data, dim_names = xml_to_dict("Dataset\\exemplars.xml", domain_mapping=domain_mapping, data=data, dimension_names=dim_names)
    domain_mapping, data, dim_names = less_dimensions(domain_mapping=domain_mapping, data=data, dim_names=dim_names)
    space.init(len(dim_names), domain_mapping, dim_names)
    concepts=form_supercategories(data)
    for name,values in zip(concepts,concepts.values()):
        if not name=='mammal':
            family_into_space(name, values, add=True)
    ci.init(dims=[6,7,8])

    ''