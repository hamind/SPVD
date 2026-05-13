img2dataset \
    --url_list .datasets/cc3m_3long_3short_1raw_captions_url.parquet \
    --input_format "parquet" \
    --url_col "Image Path" \
    --output_format "webdataset" \
    --output_folder ./datasets/scraped_cc3m \
    --processes_count 32 \
    --thread_count 64 \
    --number_sample_per_shard 5000 \
    --save_additional_columns '["raw_caption", "shortIB_captions", "longIB_captions", "shortSV_captions", "longSV_captions", "shortLLA_captions", "longLLA_captions"]'