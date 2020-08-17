import gc

import pandas as pd

ad_df = pd.read_csv("train_preliminary/ad.csv")

ad_df.replace('\\N', '0', inplace=True)

semi_ad_df = pd.read_csv("train_semi_final/ad.csv")

semi_ad_df.replace('\\N', '0', inplace=True)

user_df = pd.read_csv("train_preliminary/user.csv")

semi_user_df = pd.read_csv("train_semi_final/user.csv")

log_df = pd.read_csv("train_preliminary/click_log.csv")

semi_log_df = pd.read_csv("train_semi_final/click_log.csv")

test_ad_df = pd.read_csv("test/ad.csv")
test_ad_df.replace('\\N', '0', inplace=True)

test_log_df = pd.read_csv("test/click_log.csv")

log_wide = pd.merge(log_df, ad_df, how='inner', on="creative_id")

semi_log_wide = pd.merge(semi_log_df, semi_ad_df, how='inner', on="creative_id")

test_log_wide = pd.merge(test_log_df, test_ad_df, how='inner', on="creative_id")

gc.collect()
all_log_df = pd.concat([log_wide, semi_log_wide, test_log_wide], axis=0, ignore_index=True)

all_log_df.to_pickle('all_log_df.pkl')

all_log_df["creative_id"] = all_log_df["creative_id"].astype(str)
creative_id_docs = all_log_df.sort_values('time', ascending=True).groupby(by='user_id')["creative_id"].apply(
    list).reset_index()

all_log_df["advertiser_id"] = all_log_df["advertiser_id"].astype(str)
advertiser_id_docs = all_log_df.sort_values('time', ascending=True).groupby(by='user_id')["advertiser_id"].apply(
    list).reset_index()

all_log_df["ad_id"] = all_log_df["ad_id"].astype(str)
ad_id_docs = all_log_df.sort_values('time', ascending=True).groupby(by='user_id')["ad_id"].apply(list).reset_index()

all_log_df["product_id"] = all_log_df["product_id"].astype(str)
product_id_docs = all_log_df.sort_values('time', ascending=True).groupby(by='user_id')["product_id"].apply(
    list).reset_index()

all_log_df["product_category"] = all_log_df["product_category"].astype(str)
product_category_docs = all_log_df.sort_values('time', ascending=True).groupby(by='user_id')["product_category"].apply(
    list).reset_index()

all_log_df["industry"] = all_log_df["industry"].astype(str)
industry_docs = all_log_df.sort_values('time', ascending=True).groupby(by='user_id')["industry"].apply(
    list).reset_index()

click_times_docs = all_log_df.sort_values('time', ascending=True).groupby(by='user_id')["click_times"].apply(
    list).reset_index()

all_log_agg_wide = advertiser_id_docs.merge(creative_id_docs, how="inner", on="user_id")
all_log_agg_wide = all_log_agg_wide.merge(ad_id_docs, how="inner", on="user_id")
all_log_agg_wide = all_log_agg_wide.merge(product_id_docs, how="inner", on="user_id")
all_log_agg_wide = all_log_agg_wide.merge(product_category_docs, how="inner", on="user_id")
all_log_agg_wide = all_log_agg_wide.merge(industry_docs, how="inner", on="user_id")
all_log_agg_wide = all_log_agg_wide.merge(click_times_docs, how="inner", on="user_id")

print(all_log_agg_wide)
all_log_agg_wide.to_pickle('all_log_agg_wide.pkl')


def convert_id_df_to_doc(df, key_str):
    id_list = df[key_str]
    click_times = df['click_times']
    doc = []
    if len(id_list) == len(click_times):
        for idx in range(len(click_times)):
            for click_time in range(click_times[idx]):
                doc.append(id_list[idx])
    return ' '.join(doc)


def convert_id_df_to_doc_list(df, key_str):
    id_list = df[key_str]
    click_times = df['click_times']
    doc = []
    if len(id_list) == len(click_times):
        for idx in range(len(click_times)):
            for click_time in range(click_times[idx]):
                doc.append(id_list[idx])
    return doc


creative_id_agg_docs = all_log_agg_wide.apply(lambda x: convert_id_df_to_doc_list(x, "creative_id"),
                                              axis=1).reset_index()

advertiser_id_agg_docs = all_log_agg_wide.apply(lambda x: convert_id_df_to_doc_list(x, "advertiser_id"),
                                                axis=1).reset_index()

ad_id_agg_docs = all_log_agg_wide.apply(lambda x: convert_id_df_to_doc_list(x, "ad_id"), axis=1).reset_index()

product_id_agg_docs = all_log_agg_wide.apply(lambda x: convert_id_df_to_doc_list(x, "product_id"), axis=1).reset_index()

product_category_agg_docs = all_log_agg_wide.apply(lambda x: convert_id_df_to_doc_list(x, "product_category"),
                                                   axis=1).reset_index()

industry_agg_docs = all_log_agg_wide.apply(lambda x: convert_id_df_to_doc_list(x, "industry"), axis=1).reset_index()

all_log_agg_wide["creative_id_agg_docs"] = creative_id_agg_docs[0]
all_log_agg_wide["advertiser_id_agg_docs"] = advertiser_id_agg_docs[0]
all_log_agg_wide["ad_id_agg_docs"] = ad_id_agg_docs[0]
all_log_agg_wide["product_id_agg_docs"] = product_id_agg_docs[0]
all_log_agg_wide["product_category_agg_docs"] = product_category_agg_docs[0]
all_log_agg_wide["industry_agg_docs"] = industry_agg_docs[0]
print(all_log_agg_wide.iloc[2]['creative_id_agg_docs'])

all_log_agg_wide.to_pickle('all_log_agg_wide_semi_input_1.pkl')
