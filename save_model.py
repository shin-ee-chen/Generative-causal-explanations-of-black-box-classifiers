
if __name__ == '__main__':
    
    model = MNIST_CVAE.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)
    
    test_result = trainer.test(
        model, test_dataloaders=test_loader, verbose=True)
    
    # Save pretrained models
    gce_path = './pretrained_models/'+ args.log_dir + '/'

    torch.save(model, os.path.join(gce_path,'model.pt'))