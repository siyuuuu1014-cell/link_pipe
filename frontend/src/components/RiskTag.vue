<script setup lang="ts">
import { computed } from 'vue'
const props = defineProps<{ level: string }>()
const label = computed(() => ({ high: '高风险', medium: '中风险', none: '未发现明显异常' }[props.level] || props.level))
const type = computed(() => ({ high: 'danger', medium: 'warning', none: 'success' }[props.level] || 'info'))
</script>

<template><el-tag :type="type as any" effect="dark" round>{{ label }}</el-tag></template>
