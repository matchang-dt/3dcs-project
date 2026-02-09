import os
from datetime import datetime
import hydra
from hydra.utils import instantiate
import lpips
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run_name = f"{cfg.wandb.project}_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    os.makedirs(f"runs/{run_name}", exist_ok=True)

    if cfg.wandb.id is None:
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config=config_dict,
        )
    else: # resume previous training
        wandb.init(
            project=cfg.wandb.project,
            id=cfg.wandb.id,
            resume="allow",
            config=config_dict,
        )

    model = instantiate(cfg.model)
    train_dataset = instantiate(cfg.train_dataset)
    val_dataset = instantiate(cfg.val_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.val_batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    lpip_criterion = lpips.LPIPS(net='vgg')
    l2_criterion = nn.MSELoss()

    pbar = tqdm(train_dataloader, desc="Training")
    is_gt_logged = False

    for epoch in range(cfg.epochs):
        losses = []
        l2_losses = []
        lpips_losses = []
        epoch_loss = 0
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch+1}/{cfg.epochs}")
            optimizer.zero_grad()

            gen_images = model(batch)

            target = batch.target
            l2_loss = l2_criterion(gen_images, target.images)
            lpips_loss = lpip_criterion(gen_images, target.images)
            loss = l2_loss + 0.05*lpips_loss

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            l2_losses.append(l2_loss.item())
            lpips_losses.append(lpips_loss.item())
            epoch_loss += loss.item()

            # display and log training loss
            if i % cfg.log_interval == 0:
                avg_loss = sum(losses)/len(losses)
                avg_l2_loss = sum(l2_losses)/len(l2_losses)
                avg_lpips_loss = sum(lpips_losses)/len(lpips_losses)
                wandb.log({
                    "loss": avg_loss, 
                    "l2_loss": avg_l2_loss, 
                    "lpips_loss": avg_lpips_loss
                })
                pbar.set_postfix(
                    loss=avg_loss, 
                    l2_loss=avg_l2_loss, 
                    lpips_loss=avg_lpips_loss
                )
                losses = []
                l2_losses = []
                lpips_losses = []

        # evaluate on validation set
        with torch.no_grad():
            val_loss = 0
            is_first_batch = True
            for batch in val_dataloader:
                gen_images = model(batch)
                target = batch.target
                l2_loss = l2_criterion(gen_images, target.images)
                lpips_loss = lpip_criterion(gen_images, target.images)
                loss = l2_loss + 0.05*lpips_loss
                val_loss += loss.item()
                # log images on a few scenes in the first batch on val set
                if is_first_batch:
                    is_first_batch = False

                    if not is_gt_logged:
                        gt_image_list = []
                        for i in range(cfg.training.num_image_log):
                            images = [wandb.Image(image) for image in target.images[i]]
                            gt_image_list += images
                        wandb.log({
                            "val_target": gt_image_list,
                        })
                        is_gt_logged = True
                    
                    gen_images = gen_images.clamp(0, 1)
                    gen_image_list = []
                    for i in range(cfg.training.num_image_log):
                        images = [wandb.Image(image) for image in gen_images[i]]
                        gen_image_list += images
                    wandb.log({
                        "val_images": gen_image_list,
                    })
            val_loss /= len(val_dataloader)

        epoch_loss /= len(train_dataloader)
        wandb.log({
            "train_loss": epoch_loss,
            "val_loss": val_loss,
        })
        # save model and log artifact
        torch.save(model.state_dict(), f"runs/{run_name}/{epoch+1}.pth")
        artifact = wandb.Artifact(f"model_{epoch+1}", type="model")
        artifact.add_reference(f"runs/{run_name}/{epoch+1}.pth")
        wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    train()