# Hypertune

`pip install git+https://github.com/kubosis/NPFL_model_optimizer.git`

## Overview - how to configurate trainer
The configuration file is a YAML-based specification that guides the hyperparameter optimization process for machine learning models using Optuna. It is divided into two main sections: `self` and `functional`.

## Configuration Structure

### Top-Level Sections
1. `self`: Defines model-specific parameters and initial configuration
2. `functional`: Specifies training, configuration, and component details

## Detailed Configuration Guide

### `self` Section
The `self` section allows you to define model-specific parameters with hyperparameter optimization capabilities.

#### Supported Constructors
- `!categorical`: Suggests a categorical (discrete) value from a list
- `!float`: Suggests a floating-point value within a range
- `!int`: Suggests an integer value within a range
- `!class`: Dynamically resolves or suggests a class
- `!eval`: Evaluates a Python expression
- `!registered`: Uses a pre-registered parameter

#### Example `self` Section
```yaml
self:
    early_stop: true  # Boolean flag for early stopping
    patience: 5  # Number of epochs to wait for improvement
    batch_size: !categorical [64, 128, 256, 512, 1024]  # Categorical batch sizes
    module:
        class: !class "{{resolve}}"  # Auto-resolve current module class
        blocks: !categorical [[2,2,2,2], [3,3,3,3], [2,3,2,3]]  # Network block configurations
        convtype: !categorical ["ConvResidual", "ConvResidualWithBottleneck"]  # Convolution type options
```

### `functional` Section
Defines optimization details for different components of the training process.

#### Subsections
1. `fit`: Training configuration
2. `configure`: Optimizer, scheduler, and loss function settings

#### Example `functional` Section
```yaml
functional:
    fit:
        epochs: 25  # Fixed number of training epochs

    configure:
        optimizer:
            class: !class "torch.optim.AdamW"  # Optimizer class
            params: "!eval:model.module.parameters(recurse=True)"  # Dynamic parameter extraction
            lr: !float [5e-5, 1e-3]  # Learning rate range
            weight_decay: !float [1e-5, 5e-4]  # Weight decay range

        scheduler:
            class: !class "torch.optim.lr_scheduler.CosineAnnealingLR"  # Learning rate scheduler
            optimizer: "^hook:optimizer"  # Reference to the configured optimizer
            T_max: !registered "T_max"  # Registered parameter
            eta_min: !eval "trial.params.get('lr', 1e-4) / 100"  # Minimum learning rate

        loss:
            class: !class "torch.nn.CrossEntropyLoss"  # Loss function
```

## Advanced Configuration Techniques

### Special Constructors

#### `!categorical`
- Suggests a value from a predefined list
- Can contain various types (integers, strings, lists)
- Example: `param: !categorical [64, 128, 256]`

#### `!float`
- Suggests a floating-point value within a specified range
- Format: `!float [min_value, max_value]`
- Example: `learning_rate: !float [1e-5, 1e-3]`

#### `!int`
- Suggests an integer value within a specified range
- Format: `!int [min_value, max_value]`
- Example: `num_layers: !int [2, 5]`

#### `!class`
- Resolves or suggests a class dynamically
- Special value `"{{resolve}}"` auto-resolves the current class
- Can specify fully qualified class path
- Example: 
  ```yaml
  optimizer: 
    class: !class ["torch.optim.Adam", "torch.optim.AdamW"]
  ```

#### `!eval`
- Evaluates a Python expression during configuration parsing
- Useful for dynamic parameter computation
- Example: `min_lr: !eval "trial.params.get('lr') / 10"`

#### `!registered`
- Uses a pre-registered parameter from the model
- Example: `T_max: !registered "T_max"`

## Post-Parsing Special Functions

### `!eval:` - Post-Construction Evaluation

The `!eval:` function allows you to evaluate a Python expression *after* the initial configuration parsing. This is particularly powerful because it means you can reference parameters or objects that have already been constructed.

#### Key Characteristics
- Evaluated after all other parameters in the current section have been parsed and constructed
- Can access model attributes, constructed objects, and trial parameters
- Provides flexibility for dynamic parameter computation

#### Examples

1. Using Model Parameters
```yaml
functional:
    configure:
        optimizer:
            class: !class "torch.optim.AdamW"
            params: "!eval:model.module.parameters(recurse=True)"
            # Accesses model.module AFTER it has been constructed
```


### `^hook:` - Reference Constructed Parameters

The `^hook:` syntax allows you to reference a parameter that has already been constructed within the same configuration section. This is particularly useful for components that depend on previously constructed objects.

#### Key Characteristics
- Works only within the same nested configuration section
- Allows you to pass a previously constructed object to another component
- Provides a way to link related configuration elements

#### Examples

1. Linking Optimizer to Scheduler
```yaml
functional:
    configure:
        optimizer:
            class: !class "torch.optim.AdamW"
            lr: !float [1e-4, 1e-3]
        
        scheduler:
            class: !class "torch.optim.lr_scheduler.CosineAnnealingLR"
            optimizer: "^hook:optimizer"  # References the previously constructed optimizer
            T_max: 10
```

2. Complex Dependency Chaining
```yaml
functional:
    configure:
        base_model:
            class: !class "SomeBaseModel"
        
        feature_extractor:
            class: !class "FeatureExtractor"
            base_model: "^hook:base_model"  # Uses the previously constructed base model
```

### Interaction Between `!eval:` and `^hook:`

These functions can be used together to create complex, dynamic configurations:

```yaml
functional:
    configure:
        optimizer:
            class: !class "torch.optim.AdamW"
            params: "!eval:model.module.parameters()"
            lr: !float [1e-4, 1e-3]
        
        scheduler:
            class: !class "torch.optim.lr_scheduler.CosineAnnealingLR"
            optimizer: "^hook:optimizer"  # References the optimizer
            T_max: !eval "len(train_dataset) // trial.params.get('batch_size', 64)"
```

## Best Practices
- Use `"!eval:"` for expressions that require post-construction context
- Use `^hook:` to link related configuration components
- Ensure that referenced objects exist in the same configuration section
- Keep expressions simple and focused
- Use categorical suggests for discrete hyperparameters
- Use float/int suggests for continuous hyperparameters
- Leverage `!eval` for complex, dynamic parameter computations
- Keep the configuration flexible and expressive

## Notes
- Not all parameters need to be optimized
- Fixed values can be used alongside hyperparameter suggestions
- The configuration supports complex, nested structures

## Limitations
- `^hook:` only works within the same nested configuration section
- `"!eval:"` is evaluated after initial parsing, so it cannot modify previously constructed objects