from typing import *
from .config import ExpConfig

import dgl
from loguru import logger
import dgl.dataloading
import torch
import torch.backends.cudnn
from tqdm import tqdm
import os

from tracegnn.data import *
from tracegnn.utils import *


from .dataset import TrainDataset
from .evaluate import evaluate
from .models.level_model import LevelModel, level_model_loss


def train_epoch(config: ExpConfig,
                loader: dgl.dataloading.GraphDataLoader,
                model: LevelModel,
                optimizer):
    # Calculate total loss per epoch
    total_loss = 0.0
    lat_loss = 0.0
    kl_loss = 0.0
    total_step = 0

    t = tqdm(loader) if config.enable_tqdm else loader

    operation_loss_dict = torch.zeros([config.DatasetParams.operation_cnt], dtype=torch.float, device=config.device)
    operation_cnt_dict = torch.zeros([config.DatasetParams.operation_cnt], dtype=torch.long, device=config.device)

    for step, graph_batch in enumerate(t):
        graph_batch = graph_batch.to(config.device)
        pred = model(graph_batch)
        node_type_loss, status_loss, latency_loss, structure_kl, latency_kl, cur_loss_dict, cur_cnt_dict = \
            level_model_loss(config, pred, graph_batch, return_detail=True)
        loss = node_type_loss + status_loss + latency_loss + config.Model.kl_weight * (structure_kl + latency_kl)

        operation_loss_dict += cur_loss_dict
        operation_cnt_dict += cur_cnt_dict

        # Set loss info
        kl_loss += (structure_kl + latency_kl).item()
        total_loss += loss.item()
        lat_loss += latency_loss.item()
        total_step += 1

        # Optimize
        optimizer.zero_grad()
        loss.backward()

        # Click global grad to avoid nan
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Show loss
        if config.enable_tqdm:
            t.set_description(f'lat:{latency_loss.item():.2f} kl:{(structure_kl + latency_kl).item():.2f} L:{loss.item():.2f}')

    return total_loss / total_step, lat_loss / total_step, kl_loss / total_step, operation_loss_dict, operation_cnt_dict


def val_epoch(config: ExpConfig,
                loader: dgl.dataloading.GraphDataLoader,
                model: LevelModel):
    # Calculate total loss per epoch
    total_loss = 0.0
    lat_loss = 0.0
    kl_loss = 0.0
    total_step = 0

    t = tqdm(loader) if config.enable_tqdm else loader

    model.eval()
    
    with torch.no_grad():
        for step, graph_batch in enumerate(t):
            graph_batch = graph_batch.to(config.device)
            pred = model(graph_batch)
            node_type_loss, status_loss, latency_loss, kl_structure, kl_latency = \
                level_model_loss(config, pred, graph_batch)
            loss = node_type_loss + status_loss + latency_loss + config.Model.kl_weight * (kl_structure + kl_latency)

            # Set loss info
            total_loss += loss.item()
            lat_loss += latency_loss.item()
            kl_loss += (kl_structure + kl_latency).item()
            total_step += 1

            # Show loss
            if config.enable_tqdm:
                t.set_description(f'lat:{latency_loss.item():.2f} L:{loss.item():.2f}')

    model.train()

    return total_loss / total_step, lat_loss / total_step, kl_loss / total_step


def trainer(config: ExpConfig,
          train_loader: dgl.dataloading.GraphDataLoader, 
          val_loader: dgl.dataloading.GraphDataLoader,
          test_loader: dgl.dataloading.GraphDataLoader = None):

    # Define model and optimizer
    model = LevelModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters())

    # Set min loss for evalation
    min_val_loss = 1e9

    # Train model
    logger.info('Start training...')
    for epoch in range(config.max_epochs):
        logger.info(f'-------------> Epoch {epoch}')
        
        # Train 1 epoch
        logger.info(f'Training...')

        total_loss, lat_loss, kl_loss, operation_loss_dict, operation_cnt_dict = train_epoch(config, train_loader, model, optimizer)
        # Set output
        logger.info(f'Train Epoch: {epoch}  Loss: {total_loss} Latency Loss: {lat_loss} KL Loss: {kl_loss}')

        # Val 1 epoch
        val_total_loss, val_lat_loss, val_kl_loss = val_epoch(config, val_loader, model)
        logger.info(f'Valid Epoch: {epoch}  Loss: {val_total_loss} Latency Loss: {val_lat_loss} KL Loss: {val_kl_loss}')

        if val_total_loss < min_val_loss:
            min_val_loss = val_total_loss

            if test_loader is not None:
                logger.info('Valid loss is smaller. Start evaluation...')
                evaluate(config, test_loader, model)

            # Save model
            logger.info('Valid loss is smaller. Model saved.')
            torch.save(model.state_dict(), 'model.pth')

    # Final evaluation
    if test_loader is not None:
        evaluate(config, test_loader, model)

    logger.info('Training finished.')
