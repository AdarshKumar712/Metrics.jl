# Julia implementation of ROUGE score
# Ref: https://github.com/google/seq2seq/blob/master/seq2seq/metrics/rouge.py

"""
    _get_ngrams(n, text)

Calcualtes n-grams. Returns a set of n-grams.

# Arguments:
 - `n`: provide which n-grams to calculate
 - `text`: An array of tokens
"""
function _get_ngrams(n, text)
    ngrams_set = Set()
    text_length = length(text)
    max_index_ngram_start = text_length - n
    for i in 1:max_index_ngram_start+1
        push!(ngrams_set, tuple(text[i:i+n-1]))
    end
    return ngrams_set
end

"""
    _split_into_words(sentences)

Splits multiple sentences into words and flattens the result
"""
function _split_into_words(sentences)
    words = []
    for i in 1:length(sentences)
        temp = split(sentences[i], " ")
        for j in temp
            push!(words,j)
        end
    end
    return words
end

"""
    _get_word_ngrams(n, sentences)

Calculates word n-grams for multiple sentences.
"""
function _get_word_ngrams(n, sentences)
    @assert length(sentences) > 0
    @assert n>0
    
    words = _split_into_words(sentences)
    return _get_ngrams(n, words)
end

"""
    _lcs(x, y)

Utility function to compute the length of the longest common subsequence (lcs) between two
strings. The implementation below uses a DP programming algorithm and runs
in O(nm) time where n = len(x) and m = len(y).
"""
function _lcs(x, y)
    n, m = length(x), length(y)
    table = Dict()
    for i in 1:n+1
        for j in 1:m+1
            if i==1 || j==1
                table[i, j] = 0
            elseif x[i-1] == y[j-1]
                table[i, j] = table[i-1, j-1] + 1
            else
                table[i, j] = max(table[i-1, j], table[i, j-1])
            end
        end
    end
    return table
end 

"""
    _len_lcs(x, y)

Returns the length of the Longest Common Subsequence between sequences x and y.
"""
function _len_lcs(x, y)
    table = _lcs(x, y)
    n, m = length(x), length(y)
    return table[n+1, m+1]
end

"""
    _recons_lcs(x, y)

Returns the Longest Subsequence between x and y.
"""
function _recon_lcs(x, y)
    i , j = length(x), length(y)
    table = _lcs(x, y)
           
    function _recon(i, j)
        if i == 1 || j == 1
            return []
        elseif x[i-1] == y[j-1]
            return push!(_recon(i - 1, j - 1) ,(x[i - 1], i))
        elseif table[i - 1, j] > table[i, j - 1]
            return _recon(i - 1, j)
        else
            return _recon(i, j - 1)
        end
    end 

    recon_tuple = tuple(map(x->x[1][1], _recon(i, j))...)
end

"""
    rouge_n(evaluated_sentences, reference_sentences; n=2)

Computes ROUGE-N of two text collections of sentences. Returns f1, precision, recall for ROUGE-N.

# Arguments:
 - `evaluated_sentences`: the sentences that have been picked by the summarizer
 - `reference_sentences`: the sentences from the referene set
 - `n`: size of ngram.  Defaults to 2.

Source: (http://research.microsoft.com/en-us/um/people/cyl/download/
  papers/rouge-working-note-v1.3.1.pdf)
"""
function rouge_n(evaluated_sentences, reference_sentences; n=2)
    if length(evaluated_sentences) <= 0 || length(reference_sentences)<=0
        throw(ArgumentError())
    end
    
    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = length(reference_ngrams)
    evaluated_count = length(evaluated_ngrams)
    
    overlapping_ngrams = intersect(evaluated_ngrams, reference_ngrams)
    overlapping_count = length(overlapping_ngrams)
    precision = 0.0
    if evaluated_count != 0
         precision = overlapping_count / evaluated_count
    end
    
    recall = 0.0
    if reference_count != 0
         recall = overlapping_count / reference_count
    end
    
    f1_score = 2.0 * ((precision * recall)) / (precision + recall + 1e-9)
    return f1_score, precision, recall
end

"""
    _f_p_r_lcs(llcs, m, n)

Computes the LCS-based F-measure score

# Arguments:
 - `llcs`: Length of LCS
 - `m`: number of words in reference summary
 - `n`: number of words in candidate summary

Source: (http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf)
"""
function _f_p_r_lcs(llcs, m, n)
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta^2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta^2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return f_lcs, p_lcs, r_lcs 
end

"""
    rouge_l_sentence_level(evaluated_sentences, reference_sentences)

Computes ROUGE-L (sentence level) of two text collections of sentences.

Calculated according to:
  R_lcs = LCS(X,Y)/m,
  P_lcs = LCS(X,Y)/n,
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
 
where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary
  
# Argumnets:
 - `evaluated_sentences`: the sentences that have been picked by the summarizer
 - `reference_sentences`: the sentences from the referene set

Source: (http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf)
"""
function rouge_l_sentence_level(evaluated_sentences, reference_sentences)
    if length(evaluated_sentences) <= 0 || length(reference_sentences)<=0
        throw(ArgumentError())
    end
    
    reference_words = _split_into_words(reference_sentences)
    evaluated_words = _split_into_words(evaluated_sentences)
    m = length(reference_words)
    n = length(evaluated_words)
    lcs = _len_lcs(evaluated_words, reference_words)
    return _f_p_r_lcs(lcs, m, n)
end

"""
    _union_lcs(evaluated_sentences, reference_sentence)

Returns LCS_u(r_i, C) which is the LCS score of the union longest common subsequence between reference sentence ri and candidate summary C.

# Arguments:
 - `evaluated_sentences`: the sentences that have been picked by the summarizer
 - `reference_sentence`: one of the sentences in the reference summaries

For example, if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8 and c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1 is “w1 w2” and the longest common subsequence of r_i and c2 is “w1 w3 w5”. The union longest common subsequence of r_i, c1, and c2 is “w1 w2 w3 w5” and LCS_u(r_i, C) = 4/5.

"""
function _union_lcs(evaluated_sentences, reference_sentence)
    if length(evaluated_sentences) <= 0
        throw(ArgumentError())
    end
    
    lcs_union = Set()
    reference_words = _split_into_words([reference_sentence])
    combined_lcs_length = 0
    for eval_s in evaluated_sentences
        evaluated_words = _split_into_words([eval_s])
        lcs = Set(_recon_lcs(reference_words, evaluated_words))
        combined_lcs_length += length(lcs)
        lcs_union = union(lcs, lcs_union)
    end
    
    union_lcs_count = length(lcs_union)
    union_lcs_value = union_lcs_count / combined_lcs_length
    return union_lcs_value
end

"""
    rouge_l_summary_level(evaluated_sentences, reference_sentences)

Computes ROUGE-L (summary level) of two text collections of sentences.

Calculated according to:
  R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
  P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

where:
  SUM(i,u) = SUM from i through u
  u = number of sentences in reference summary
  C = Candidate summary made up of v sentences
  m = number of words in reference summary
  n = number of words in candidate summary
  
# Arguments:
 - `evaluated_sentences`: the sentences that have been picked by the summarizer
 - `reference_sentence`: the sentences in the reference summaries

Source: (http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf)
  
"""
function rouge_l_summary_level(evaluated_sentences, reference_sentences)
    if length(evaluated_sentences) <= 0 || length(reference_sentences)<=0
        throw(ArgumentError())
    end
    m = length(_split_into_words(reference_sentences))
    n = length(_split_into_words(evaluated_sentences))
    union_lcs_sum_across_all_references = 0
    for ref_s in reference_sentences
        union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences, ref_s)
    end
    return _f_p_r_lcs(union_lcs_sum_across_all_references, m, n)
end

"""
    rouge(hypotheses, references)

Calculates average rouge scores for a list of hypotheses and references.
"""
function rouge(hypotheses, references)
    rouge_1 = [rouge_n([hyp],[ref], n=1) for (hyp, ref) in zip(hypotheses, references)]
    rouge_1_f, rouge_1_p, rouge_1_r = 0.0,0.0,0.0
    for i in 1:length(rouge_1)
        rouge_1_f+=rouge_1[i][1]
        rouge_1_p+=rouge_1[i][2]
        rouge_1_r+=rouge_1[i][3]
    end
    n = length(rouge_1)
    rouge_1_f/= n
    rouge_1_p/= n
    rouge_1_r/= n
    
    rouge_2 = [rouge_n([hyp],[ref],n=2) for (hyp, ref) in zip(hypotheses, references)]
    rouge_2_f, rouge_2_p, rouge_2_r = 0.0,0.0,0.0
    for i in 1:length(rouge_2)
        rouge_2_f+=rouge_2[i][1]
        rouge_2_p+=rouge_2[i][2]
        rouge_2_r+=rouge_2[i][3]
    end
    n = length(rouge_1)
    rouge_2_f/= n
    rouge_2_p/= n
    rouge_2_r/= n
    
    rouge_l = [rouge_l_sentence_level([hyp],[ref]) for (hyp, ref) in zip(hypotheses, references)]
    rouge_l_f, rouge_l_p, rouge_l_r = 0.0,0.0,0.0
    for i in 1:length(rouge_l)
        rouge_l_f+=rouge_l[i][1]
        rouge_l_p+=rouge_l[i][2]
        rouge_l_r+=rouge_l[i][3]
    end
    n = length(rouge_l)
    rouge_l_f/= n
    rouge_l_p/= n
    rouge_l_r/= n

    return OrderedDict(
      "rouge_1 / f_score"=> rouge_1_f,
      "rouge_1 / r_score"=> rouge_1_r,
      "rouge_1 / p_score"=> rouge_1_p,
      "rouge_2 / f_score"=> rouge_2_f,
      "rouge_2 / r_score"=> rouge_2_r,
      "rouge_2 / p_score"=> rouge_2_p,
      "rouge_l / f_score"=> rouge_l_f,
      "rouge_l / r_score"=> rouge_l_r,
      "rouge_l / p_score"=> rouge_l_p,)
    
end
