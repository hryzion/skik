import os

batches = [
    "719ca5f3-bdd2-41c0-b227-dad58995e76d",
    '36678f29-129a-40ef-b197-a2c3fde6778c',
    "b2f891f3-2d68-472d-a7ac-6e7c606988b5",
    "518cf547-232e-4ae1-927f-a778899c487d",
    "121c5eb7-d5f1-4cf9-975a-a51e290d3cfd",
    "39fd85a5-c974-4bc7-8677-7cea578b997c",
    "bac4585a-0c28-4fb1-b62f-819a1e9a5922",
    "9f95c9f8-3e85-4409-a803-807656b46bb2",
    # 'c41c0568-a8c6-4eef-ae67-cd73a7aa7b1b',
    # "a3a08f99-7afd-4a27-96c1-0125e847109f",
    # "25e56cfd-4764-470a-9f14-5fb04abe9dbb",
    # "d0716e06-2e4a-4b93-8008-aa721a936062",
    # "43422e7f-3bb8-44f8-add9-f11cfec2480c",
    # "34a2d72b-551d-48ea-aeef-5428fb2002e0",
    # "85332070-5fa0-412e-bc3a-028540bf5912",
    # "379c9edf-f1ad-40d7-95cc-8e96a928a2b4",
    # "30ffb982-3e7d-4163-8790-f2817cd78680",
    # "a6ea0861-0de4-4767-bb3a-ea475bc92628",
    # "2b5ad0b1-95d2-42cb-be8a-668f8845fd5c",
    # "e5871826-9cb3-47c9-b3f9-7d38b9cbfd21"
]

for scene_token in batches:
    os.system(f'python Seeker.py --val --width 128 --height 96 --scene {scene_token} --epoch 1000 --cuda 0')