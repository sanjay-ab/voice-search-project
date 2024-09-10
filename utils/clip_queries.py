import soundfile as sf

# change as needed
TRANSCRIPT_PATH = "data"
AUDIO_PATH = "data"
AUDIO_OUT_PATH = "data"


def get_query_word_timings(lang):
    """ Get a list of the query_filename, source_doc, keyword, start_time, end_time to be used
    to clip keywords from the original audio

    Args:
        lang (str): language of queries

    Returns:
        list[str]: list of the query_filename, source_doc, keyword, start_time, end_time
    """
    # ref file
    ref_list = []
    with open(f"{TRANSCRIPT_PATH}/{lang}/analysis/ref_of_queries_in_docs.txt", "r") as f:
        for line in f:
            line_lst = line.split()
            query_filename = line_lst[0].strip()
            keyword = line_lst[1].strip()
            ref_list.append((query_filename, keyword))

    # query word timing file
    timings_list = []
    with open(f"{TRANSCRIPT_PATH}/{lang}/analysis/queries_times.txt", "r") as f:
        for line in f:
            source_doc, keyword, start_time, end_time = line.split()
            start_time = round(float(start_time.strip()), 2)
            end_time = round(float(end_time.strip()), 2)
            timings_list.append((source_doc, keyword, start_time, end_time))

    # combine info
    query_names_with_times = []
    for ref_entry, timings_entry in zip(ref_list, timings_list):
        query_filename, keyword = ref_entry
        source_doc, keyword2, start_time, end_time = timings_entry
        if keyword != keyword2:
            print("keyword doesn't match")
            print(keyword, keyword2)
            exit(0)
        query_names_with_times.append((query_filename, source_doc, keyword, start_time, end_time))
    return query_names_with_times

def clip_query(lang, source_file, out_file, start_time, end_time):
    """Using forced alignment timings, cut keyword out of source audio.

    Args:
        lang (str): language of data
        source_file (str): name of original audio file
        out_file (str): name of clipped query file
        start_time (float): start time of keyword from forced alignment
        end_time (float): end time of keyword from forced alignment
    """

    # change this path as needed
    y, sr = sf.read(f"{AUDIO_PATH}/{lang}/all_data/{source_file}.wav")  

    start_sample = int(sr * start_time)
    end_sample = int(sr * end_time)

    query = y[start_sample:end_sample + 1]  # include last sample
    outpath = f"{AUDIO_OUT_PATH}/{lang}/queries/{out_file}.wav"
    sf.write(outpath, query, samplerate=sr)

def clip_queries_bulk(lang):
    """Clip queries for all queries in a given language.

    Args:
        lang (str): language of queries
    """
    query_names_with_times = get_query_word_timings(lang)
    for query_filename, source_doc, _, start_time, end_time in query_names_with_times:
        print(f"Clipping {query_filename} from {source_doc}")
        clip_query(lang, source_doc, query_filename, start_time, end_time)


if __name__ == "__main__":
    # print(get_query_word_timings("gujarati")[:10])
    clip_queries_bulk("gujarati")
