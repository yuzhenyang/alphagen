import subprocess

def get_gpu_processes():
    # 执行nvidia-smi命令并获取输出
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid', '--format=csv,noheader'])

    # 将输出按行分割并去除空行
    lines = result.decode('utf-8').split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    # 解析输出，获取进程ID和使用的GPU ID
    gpu_processes = {}
    for line in lines:
        parts = line.split(',')
        if len(parts) == 3:
            pid, process_name, gpu_uuid = parts
            gpu_uuid = gpu_uuid.strip()
            gpu_processes.setdefault(gpu_uuid, []).append((int(pid), process_name))
    
    return gpu_processes


def get_available_gpus():
    # 执行nvidia-smi -L命令并获取输出
    result = subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8')

    # 将输出按行分割并去除空行
    lines = result.strip().split('\n')

    gpu_info = {}
    for line in lines:
        parts = line.strip().split(' ')
        # 提取GPU索引和UUID
        gpu_index = int(parts[1].split(':')[0])
        gpu_uuid = parts[-1]
        gpu_info[gpu_uuid.strip(')')] = gpu_index
    
    return gpu_info


def get_nousing_gpus():
    gpu_processes = get_gpu_processes()
    available_gpus = get_available_gpus()
    for gpu_id, processes in gpu_processes.items():
        # print(gpu_id)
        if gpu_id in available_gpus:
            del available_gpus[gpu_id]
    
    return list(available_gpus.values())


if __name__ == "__main__":
    print(get_nousing_gpus())


    gpu_processes = get_gpu_processes()
    available_gpus = get_available_gpus()
    
    for uuid, index in available_gpus.items():
        print(f"Available GPU {index}: {uuid}")

    for k, v in gpu_processes.items():
        print(f"Inusing GPU UUID: {k}, process: {v}")

    for gpu_id, processes in gpu_processes.items():
        # print(gpu_id)
        gpu_index = available_gpus[gpu_id]
        print(f"GPU ID: {gpu_id}, Index: {gpu_index}")
        for pid, process_name in processes:
            print(f"Process ID: {pid}, Process Name: {process_name}")

