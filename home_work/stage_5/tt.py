from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import gc

def prepare_test_data(test_data, tokenizer, max_length=512):
    """
    准备测试数据，进行tokenization
    
    Args:
        test_data: List of dictionaries containing text data
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with tokenized inputs
    """
    # 获取文本字段名（假设是第一个字段）
    text_field = list(test_data[0].keys())[0] if test_data else "text"
    
    texts = [item[text_field] for item in test_data]
    
    # Tokenize the texts
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return tokenized

def create_batch_generator(tokenized_data, batch_sizes=[4, 8, 16, 32]):
    """
    生成不同batch size的数据批次
    
    Args:
        tokenized_data: Dictionary with tokenized inputs
        batch_sizes: List of batch sizes to test
    
    Yields:
        tuple: (batch_size, batch_inputs)
    """
    input_ids = tokenized_data['input_ids']
    attention_mask = tokenized_data.get('attention_mask')
    
    total_samples = len(input_ids)
    
    for batch_size in batch_sizes:
        print(f"\n准备测试 batch_size={batch_size}")
        
        # 计算需要多少个批次
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            
            batch_input_ids = input_ids[start_idx:end_idx]
            batch_attention_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
            
            batch = {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask
            }
            
            yield batch_size, batch, batch_idx, num_batches

def test_throughput(model, tokenizer, test_data, batch_sizes=[4, 8, 16, 32], max_length=512):
    """
    测试模型在不同batch size下的吞吐量
    
    Args:
        model: Loaded HuggingFace model
        tokenizer: Loaded HuggingFace tokenizer
        test_data: List of test samples
        batch_sizes: List of batch sizes to test
        max_length: Maximum sequence length
    """
    print("开始准备测试数据...")
    tokenized_data = prepare_test_data(test_data, tokenizer, max_length)
    print(f"数据已准备完成，总共 {len(tokenized_data['input_ids'])} 个样本")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*50}")
        print(f"开始测试 batch_size = {batch_size}")
        print(f"{'='*50}")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        batch_generator = create_batch_generator(
            {'input_ids': tokenized_data['input_ids'], 
             'attention_mask': tokenized_data['attention_mask']}, 
            [batch_size]
        )
        
        total_time = 0
        processed_samples = 0
        batch_count = 0
        
        for current_batch_size, batch, batch_idx, num_batches in batch_generator:
            # Move batch to GPU if available
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) if batch['attention_mask'] is not None else None
            
            # Prepare model inputs
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            # Measure inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():  # Disable gradient computation for inference
                outputs = model(**model_inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_time += batch_time
            batch_count += 1
            processed_samples += len(input_ids)
            
            print(f"Batch {batch_idx + 1}/{num_batches}: Processed {len(input_ids)} samples in {batch_time:.4f}s")
            
            # Optional: Break after processing a few batches for quick testing
            # if batch_idx >= 10:  # Remove this for full testing
            #     break
        
        # Calculate throughput metrics
        avg_time_per_batch = total_time / batch_count if batch_count > 0 else 0
        total_samples_processed = processed_samples
        throughput_samples_per_second = total_samples_processed / total_time if total_time > 0 else 0
        throughput_batches_per_second = batch_count / total_time if total_time > 0 else 0
        
        results[batch_size] = {
            'avg_time_per_batch': avg_time_per_batch,
            'total_time': total_time,
            'total_samples_processed': total_samples_processed,
            'throughput_samples_per_second': throughput_samples_per_second,
            'throughput_batches_per_second': throughput_batches_per_second,
            'batch_count': batch_count
        }
        
        print(f"\nBatch Size {batch_size} Results:")
        print(f"  Average time per batch: {avg_time_per_batch:.4f}s")
        print(f"  Total processing time: {total_time:.4f}s")
        print(f"  Total samples processed: {total_samples_processed}")
        print(f"  Throughput (samples/sec): {throughput_samples_per_second:.2f}")
        print(f"  Throughput (batches/sec): {throughput_batches_per_second:.2f}")
    
    return results

# Example usage:
def main():
    # Your test data
    test_data = [
        {"text": "Hello, how are you today?"},
        {"text": "What is the weather like?"},
        {"text": "Tell me about artificial intelligence."},
        {"text": "Explain quantum computing."},
        {"text": "What is the capital of France?"},
        {"text": "How do neural networks work?"},
        {"text": "Write a short poem about spring."},
        {"text": "What is machine learning?"},
        # Add more test samples here...
    ]
    
    # Load your model and tokenizer
    model_name = "gpt2"  # Replace with your model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Model loaded on device: {device}")
    
    # Test different batch sizes
    batch_sizes = [4, 8, 16, 32]
    
    results = test_throughput(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        batch_sizes=batch_sizes,
        max_length=512
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF THROUGHPUT TESTS")
    print("="*60)
    for batch_size, metrics in results.items():
        print(f"Batch Size {batch_size:2d}: {metrics['throughput_samples_per_second']:6.2f} samples/sec "
              f"({metrics['throughput_batches_per_second']:4.2f} batches/sec)")

if __name__ == "__main__":
    main()