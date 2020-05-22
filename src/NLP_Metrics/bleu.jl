# Julia Implementation of BLEU and Smooth BLEU score
# ref: https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py#L56

# Example: bleu_score([["apple is apple"]], ["apple is appl"])

"""
    get_ngrams(segment, max_order)

Extracts all n-grams upto a given maximum order from an input segment. Returns the counter containing all n-grams upto max_order in segment
with a count of how many times each n-gram occurred.

# Arguments 
 - `segment`: text segment from which n-grams will be extracted.
 - `max_order`: maximum length in tokens of the n-grams returned by this methods.

"""
function get_ngrams(segment, max_order)
    ngrams_count = OrderedDict()
    for order in 1:max_order
        for i in 1: (length(segment) - order+1)
            ngram = tuple(segment[i:i+order-1]...)
            if (ngram) in keys(ngrams_count)
                ngrams_count[ngram] += 1
            else 
                ngrams_count[ngram] = 1
            end
        end
    end
    return ngrams_count
end

"""
    bleu_score(reference_corpus, translation_corpus; max_order=4, smooth=false)

Computes BLEU score of translated segments against one or more references. Returns the `BLEU score`, `n-gram precisions`, `brevity penalty`, 
geometric mean of n-gram precisions, translation_length and  reference_length

# Arguments
 - `reference_corpus`: list of lists of references for each translation. Each reference should be tokenized into a list of tokens.
 - `translation_corpus`: list of translations to score. Each translation should be tokenized into a list of tokens.
 - `max_order`: maximum n-gram order to use when computing BLEU score. 
 - `smooth=false`: whether or not to apply. Lin et al. 2004 smoothing.

"""
function bleu_score(reference_corpus, translation_corpus; max_order=4, smooth=false)
    matches_by_order = zeros(max_order)
    possible_matches_by_order = zeros(max_order)
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus)
        reference_length += min([length(r) for r in references]...)
        translation_length += length(translation)
        merged_ref_ngram_counts = OrderedDict()
        for reference in references
          ref_ngrams = get_ngrams(reference, max_order)
          keys_union = union(keys(merged_ref_ngram_counts), keys(ref_ngrams)) 
          for key in keys_union
              try (b[key])
                  try (ref_ngrams[key])
                      merged_ref_ngram_counts[key] = max(merged_ref_ngram_counts[key], ref_ngrams[i])
                  catch error
                      continue
                  end
              catch error
                  merged_ref_ngram_counts[key] = ref_ngrams[key]
              end
           end
        end
        # print(length(merged_ref_ngram_counts),"\n")
        translation_ngram_counts = get_ngrams(translation, max_order)
        overlap = OrderedDict()
        keys_union = intersect(keys(merged_ref_ngram_counts), keys(translation_ngram_counts))
        for key in keys_union
                 overlap[key] = min(translation_ngram_counts[key],  merged_ref_ngram_counts[key])
        end
        for key in overlap
            matches_by_order[length(key[1])] += key[2]
        end
        for order in 1:max_order
            possible_matches = length(translation) - order + 1
            if possible_matches > 0
                possible_matches_by_order[order] += possible_matches
            end
        end
    end
    precisions = zeros(max_order)
    for i in 1:max_order
        if smooth
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        else 
            if possible_matches_by_order[i]>0
                precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order[i])
            end
        end
    end

    geo_mean = 0.0
    if min(precisions...) > 0
       p_log_sum = sum(log.(precisions)) / max_order
       geo_mean = exp(p_log_sum)
    end 
     
    ratio =  translation_length / reference_length 
    bp = 1.0
    if ratio <1.0
       bp = exp(1 - 1 /ratio)
    end
    
    bleu = geo_mean * bp
    return bleu, precisions, bp, geo_mean, translation_length, reference_length
end
