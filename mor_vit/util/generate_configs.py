from omegaconf import OmegaConf
import os

def generate_configs(base_config_path, output_dir, variants):
    base_cfg = OmegaConf.load(base_config_path)
    os.makedirs(output_dir, exist_ok=True)

    for variant in variants:
        cfg = OmegaConf.create(base_cfg)
        for key, value in variant.items():
            OmegaConf.update(cfg, key, value)
        output_path = os.path.join(output_dir, f"{variant.get('name', 'config')}.yaml")
        with open(output_path, 'w') as f:
            OmegaConf.save(cfg, f)
        print(f"Generated config at {output_path}")

if __name__ == "__main__":
    base_config = "/content/mor_vit/conf/train/example.yaml"
    output_dir = "/content/mor_vit/conf/train/generated"
    variants = [
        {"name": "mor_vit_small", "model": "mor_vit", "per_device_train_batch_size": 64},
        {"name": "vit_baseline", "model": "vit", "per_device_train_batch_size": 64}
    ]
    generate_configs(base_config, output_dir, variants)