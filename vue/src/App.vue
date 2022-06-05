<template>
  <div class="container">
    <div class="header">
      <div class="header-left">
      <span class="icon">
        <img class="logo" src="./img/small-logo.svg" style="border-style: none;

        padding: 13px;
        border-radius: 12px;">
      </span>
        <p class="name">Цифровой таможенник <br>
          <span style="font-size:0.7rem; font-family: sans-serif;">Автоматизированная система определения кода ТН ВЭД ЕАЭС</span>
        </p>

      </div>
      <div class="header-right">
        <button class="mode-switch" title="Switch Theme">
          <svg class="moon" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" width="24" height="24" viewBox="0 0 24 24">
            <defs></defs>
            <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"></path>
          </svg>
        </button>
        <!-- <button class="add-btn" title="Add New codes">
          <svg class="btn-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="feather feather-plus">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" /></svg>
        </button> -->
        <button class="notification-btn">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-bell">
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
            <path d="M13.73 21a2 2 0 0 1-3.46 0" /></svg>
        </button>
        <button class="profile-btn">
          <img src="https://crypto.ru/wp-content/plugins/q-auth/assets/img/default-user.png" />
          <span>Имя Фамилия</span>
        </button>
      </div>
      <button class="messages-btn">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-message-circle">
          <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" /></svg>
      </button>
    </div>
    <div class="content">
      <div class="sidebar">
        <a href="" class="sidebar-link active">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-search">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
            <polyline points="9 22 9 12 15 12 15 22" /></svg>
        </a>
        <a href="" class="sidebar-link">
          <svg class="link-icon feather feather-pie-chart" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24">
            <defs />
            <path d="M21.21 15.89A10 10 0 118 2.83M22 12A10 10 0 0012 2v10z" />
          </svg>
        </a>
        <a href="" class="sidebar-link">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-calendar">
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
            <line x1="16" y1="2" x2="16" y2="6" />
            <line x1="8" y1="2" x2="8" y2="6" />
            <line x1="3" y1="10" x2="21" y2="10" /></svg>
        </a>
        <a href="" class="sidebar-link">
          <svg class="link-icon feather feather-settings" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24">
            <defs />
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
          </svg>
        </a>
      </div>
      <div class="codes-section">
        <div class="codes-section-header">
          <p>Поиск кода ТН ВЭД</p>
          <!-- <p class="time">Декабрь, 12</p> -->
        </div>
        <div class="search-area-228" style="display: grid; grid-template-columns: repeat(2, auto); grid-gap: 2rem; align-content: center; margin-bottom: 32px;">
          <div class="search-wrapper">
            <input v-model="request" class="search-input" type="text" placeholder="Введите ключевые слова или код товара" style="font-family: 'Kumbh Sans', sans-serif;">
            <button @click="search(request)" style="background: none; outline: none; border: none">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" class="feather feather-search" viewBox="0 0 24 24">
                <defs></defs>
                <circle cx="11" cy="11" r="8"></circle>
                <path d="M21 21l-4.35-4.35"></path>
              </svg>
            </button>
          </div>
          <button class="upload-btn search-wrapper" alt="загрузить">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-upload"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          </button>
        </div>
        <!-- <div class="codes_filter">
          <div class=" filter_group group_1 " >Насос</div>
          <div class=" filter_group group_2"  >Кабель</div>
          <div class=" filter_group group_3"  >Ложка</div>
          <div class="filter_group group_4"  >Бутылка</div>

        </div> -->
        <div class="codes-section-line">
          <div class="codes-status">
<!--            <div class="item-status">-->
<!--              <span class="status-type">Кол-во совпадений: </span>-->
<!--              <span class="status-number">14</span>-->
<!--            </div>-->
            <!-- <div class="item-status">
              <span class="status-number">24</span>
              <span class="status-type">Upcoming</span>
            </div>
            <div class="item-status">
              <span class="status-number">62</span>
              <span class="status-type">Total codess</span>
            </div> -->
          </div>
          <div class="view-actions">
            <button class="view-btn list-view" title="List View">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-list">
                <line x1="8" y1="6" x2="21" y2="6" />
                <line x1="8" y1="12" x2="21" y2="12" />
                <line x1="8" y1="18" x2="21" y2="18" />
                <line x1="3" y1="6" x2="3.01" y2="6" />
                <line x1="3" y1="12" x2="3.01" y2="12" />
                <line x1="3" y1="18" x2="3.01" y2="18" /></svg>
            </button>
            <button class="view-btn grid-view active" title="Grid View">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-grid">
                <rect x="3" y="3" width="7" height="7" />
                <rect x="14" y="3" width="7" height="7" />
                <rect x="14" y="14" width="7" height="7" />
                <rect x="3" y="14" width="7" height="7" /></svg>
            </button>
          </div>

        </div>
        <div v-if="show.cards" class="codes-boxes jsGridView">
          <div class="codes-box-wrapper">
            <div class="codes-box" >
              <div class="codes-box-header">
                <!-- <span>December 10, 2020</span> -->
                <div class="more-wrapper">
                  <button class="codes-btn-more">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-more-vertical">
                      <circle cx="12" cy="12" r="1" />
                      <circle cx="12" cy="5" r="1" />
                      <circle cx="12" cy="19" r="1" /></svg>
                  </button>
                </div>
              </div>
              <div class="codes-box-content-header">
                <p class="box-content-header">{{cods[0].number}}</p>
              </div>
              <div class="description">
<!--                <p>Приборы электронагревательные для приготовления кофе или чая</p>-->
              </div>
              <div class="box-progress-wrapper">
                <p class="box-progress-header">Вероятность</p>
                <div class="box-progress-bar">
                  <span class="box-progress" :style="'width:' + cods[0].p + '%'"></span>
                </div>
                <p class="box-progress-percentage">{{cods[0].p}}%</p>
              </div>

            </div>

          </div>
          <div class="codes-box-wrapper">
            <div class="codes-box" >
              <div class="codes-box-header">
                <!-- <span>December 10, 2020</span> -->
                <div class="more-wrapper">
                  <button class="codes-btn-more">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-more-vertical">
                      <circle cx="12" cy="12" r="1" />
                      <circle cx="12" cy="5" r="1" />
                      <circle cx="12" cy="19" r="1" /></svg>
                  </button>
                </div>
              </div>
              <div class="codes-box-content-header">
                <p class="box-content-header">{{cods[1].number}}</p>
              </div>
              <div class="description">
<!--                <p>Кофеварки и другие приспособления для приготовления кофе и других горячих напитков, за исключением машин и оборудования, используемых в бытовых целях</p>-->
              </div>
              <div class="box-progress-wrapper">
                <p class="box-progress-header">Вероятность</p>
                <div class="box-progress-bar">
                  <span class="box-progress" :style="'width:' + cods[1].p + '%'"></span>
                </div>
                <p class="box-progress-percentage">{{cods[1].p}}%</p>
              </div>

            </div>

          </div>

          <div class="codes-box-wrapper">
            <div class="codes-box" >
              <div class="codes-box-header">
                <!-- <span>December 10, 2020</span> -->
                <div class="more-wrapper">
                  <button class="codes-btn-more">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-more-vertical">
                      <circle cx="12" cy="12" r="1" />
                      <circle cx="12" cy="5" r="1" />
                      <circle cx="12" cy="19" r="1" /></svg>
                  </button>
                </div>
              </div>
              <div class="codes-box-content-header">
                <p class="box-content-header">{{cods[2].number}}</p>
              </div>
              <div class="description">
<!--                <p>Измельчители пищевых продуктов и миксеры; соковыжималки для фруктов или овощей</p>-->
              </div>
              <div class="box-progress-wrapper">
                <p class="box-progress-header">Вероятность</p>
                <div class="box-progress-bar">
                  <span class="box-progress" :style="'width:' + cods[2].p + '%'"></span>
                </div>
                <p class="box-progress-percentage">{{cods[2].p}}%</p>
              </div>

            </div>

          </div>
<!--          <div class="codes-box-wrapper">-->
<!--            <div class="codes-box" >-->
<!--              <div class="codes-box-header">-->
<!--                &lt;!&ndash; <span>December 10, 2020</span> &ndash;&gt;-->
<!--                <div class="more-wrapper">-->
<!--                  <button class="codes-btn-more">-->
<!--                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-more-vertical">-->
<!--                      <circle cx="12" cy="12" r="1" />-->
<!--                      <circle cx="12" cy="5" r="1" />-->
<!--                      <circle cx="12" cy="19" r="1" /></svg>-->
<!--                  </button>-->
<!--                </div>-->
<!--              </div>-->
<!--              <div class="codes-box-content-header">-->
<!--                <p class="box-content-header">{{cods[3].number}}</p>-->
<!--              </div>-->
<!--              <div class="description">-->
<!--&lt;!&ndash;                <p>Изделия прочие из пластмасс и изделия из прочих материалов товарных позиций 3901 - 3914, прочие</p>&ndash;&gt;-->
<!--              </div>-->
<!--              <div class="box-progress-wrapper">-->
<!--                <p class="box-progress-header">Вероятность</p>-->
<!--                <div class="box-progress-bar">-->
<!--                  <span class="box-progress" style="width: 3.09%; "></span>-->
<!--                </div>-->
<!--                <p class="box-progress-percentage">{{cods[3].p}}</p>-->
<!--              </div>-->

<!--            </div>-->

<!--          </div>-->
<!--          <div class="codes-box-wrapper">-->
<!--            <div class="codes-box" >-->
<!--              <div class="codes-box-header">-->
<!--                &lt;!&ndash; <span>December 10, 2020</span> &ndash;&gt;-->
<!--                <div class="more-wrapper">-->
<!--                  <button class="codes-btn-more">-->
<!--                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-more-vertical">-->
<!--                      <circle cx="12" cy="12" r="1" />-->
<!--                      <circle cx="12" cy="5" r="1" />-->
<!--                      <circle cx="12" cy="19" r="1" /></svg>-->
<!--                  </button>-->
<!--                </div>-->
<!--              </div>-->
<!--              <div class="codes-box-content-header">-->
<!--                <p class="box-content-header">{{cods[4].number}} </p>-->
<!--              </div>-->
<!--              <div class="description">-->
<!--&lt;!&ndash;                <p>Прочие части оборудования для мойки или сушки бутылок или других емкостей; заполнения, закупорки бутылок, банок, закрывания ящиков, мешков, или других емкостей, для опечатывания их или этикетирования; для герметичной укупорки колпачками или крышками бутылок, банок и аналогичных емкостей; для упаковки или обертки товара; части оборудования для газирования напитков</p>&ndash;&gt;-->
<!--              </div>-->
<!--              <div class="box-progress-wrapper">-->
<!--                <p class="box-progress-header">Вероятность</p>-->
<!--                <div class="box-progress-bar">-->
<!--                  <span class="box-progress" style="width: 13%; "></span>-->
<!--                </div>-->
<!--                <p class="box-progress-percentage">{{cods[4].p}}</p>-->
<!--              </div>-->

<!--            </div>-->

<!--          </div>-->
<!--          <div class="codes-box-wrapper">-->
<!--            <div class="codes-box" >-->
<!--              <div class="codes-box-header">-->
<!--                &lt;!&ndash; <span>December 10, 2020</span> &ndash;&gt;-->
<!--                <div class="more-wrapper">-->
<!--                  <button class="codes-btn-more">-->
<!--                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-more-vertical">-->
<!--                      <circle cx="12" cy="12" r="1" />-->
<!--                      <circle cx="12" cy="5" r="1" />-->
<!--                      <circle cx="12" cy="19" r="1" /></svg>-->
<!--                  </button>-->
<!--                </div>-->
<!--              </div>-->
<!--              <div class="codes-box-content-header">-->
<!--                <p class="box-content-header">{{cods[5].number}}</p>-->
<!--              </div>-->
<!--              <div class="description">-->
<!--&lt;!&ndash;                <p>Модули с плоской дисплейной панелью, в том числе с сенсорным экраном, без драйверов или цепей управления, прочие</p>&ndash;&gt;-->
<!--              </div>-->
<!--              <div class="box-progress-wrapper">-->
<!--                <p class="box-progress-header">Вероятность</p>-->
<!--                <div class="box-progress-bar">-->
<!--                  <span class="box-progress" style="width: 9%; "></span>-->
<!--                </div>-->
<!--                <p class="box-progress-percentage">{{cods[5].p}}</p>-->
<!--              </div>-->
<!--            </div>-->
<!--          </div>-->
        </div>
      </div>
      <section class="loader-wrapper" :class="{'loader--active': show.loader}">
        <div class="loader-container">
          <div class="clock"></div>
        </div>
        <span style="font-family: sans-serif; font-size: 4rem; color: #fff">{{ warn }}</span>
      </section>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
export default {
  name: 'App',
  data() {
    return {
      request: '',
      show: {
        loader: false,
        cards: false
      },
      warn: '',
      cods: [
        {
          number: null,
          p: null
        },
        {
          number: null,
          p: null
        },
        {
          number: null,
          p: null
        },
        {
          number: null,
          p: null
        },
        {
          number: null,
          p: null
        },
        {
          number: null,
          p: null
        },
      ]
    }
  },
  created() {
    document.addEventListener('DOMContentLoaded', function () {
      var modeSwitch = document.querySelector('.mode-switch');

      modeSwitch.addEventListener('click', function () { document.documentElement.classList.toggle('dark');
        modeSwitch.classList.toggle('active');
      });

      var listView = document.querySelector('.list-view');
      var gridView = document.querySelector('.grid-view');
      var projectsList = document.querySelector('.codes-boxes');

      listView.addEventListener('click', function () {
        gridView.classList.remove('active');
        listView.classList.add('active');
        projectsList.classList.remove('jsGridView');
        projectsList.classList.add('jsListView');
      });

      gridView.addEventListener('click', function () {
        gridView.classList.add('active');
        listView.classList.remove('active');
        projectsList.classList.remove('jsListView');
        projectsList.classList.add('jsGridView');
      });

      document.querySelector('.messages-btn').addEventListener('click', function () {
        document.querySelector('.messages-section').classList.add('show');
      });

      document.querySelector('.messages-close').addEventListener('click', function() {
        document.querySelector('.messages-section').classList.remove('show');
      });
    });
  },
  methods: {
    search(input) {
      this.show.cards = false
      input = input.toUpperCase()
      this.show.loader = true
      axios.get('http://localhost:8000/fts', {
        params: {
          req: input,
          responseType: 'stream'
        }
      })
          .then(data => {
            let arr = data.data
            console.log(arr)
            arr.sort((a, b) => Number(b.prediction) - Number(a.prediction))
            arr = arr.filter(el => el.prediction > 0)

            console.log(arr)

            if (arr.length < 3) {
              this.warn = 'Ничего не найдено'
              setTimeout(() => {
                this.warn = ''
                this.show.loader = false
              }, 5000)
            } else {
              this.show.cards = true
              const cods = this.cods
              let sum = 0
              arr.map(el => sum += Number(el.prediction))
              for (let el in cods) {
                console.log(el)
                if (el >= arr.length) {
                  continue
                }
                cods[el].number = arr[el].code
                cods[el].p = Math.floor((Number(arr[el].prediction) / sum) * 100)
              }
              this.show.loader = false
            }
          })
          .catch(err => {
            this.warn = 'Ничего не найдено'
            console.log(err)
            setTimeout(() => {
              this.warn = ''
              this.show.loader = false
            }, 5000)
          })

    }
  }
}
</script>

<style>

</style>
