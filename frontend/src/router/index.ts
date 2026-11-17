import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: () => import('../views/DashboardView.vue'), meta: { title: '系统总览' } },
    { path: '/detect/image', component: () => import('../views/ImageDetectionView.vue'), meta: { title: '图片检测' } },
    { path: '/detect/video', component: () => import('../views/VideoDetectionView.vue'), meta: { title: '视频检测' } },
    { path: '/metrics', component: () => import('../views/MetricsView.vue'), meta: { title: '模型指标' } },
    { path: '/history', component: () => import('../views/HistoryView.vue'), meta: { title: '历史记录' } },
    { path: '/reports/:id', component: () => import('../views/ReportView.vue'), meta: { title: '检测报告' } },
  ],
})

router.afterEach((to) => {
  document.title = `${String(to.meta.title)}｜管道泄漏智能检测系统`
})

export default router
